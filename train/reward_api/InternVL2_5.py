"""Reward-API server: scores multiple-choice video QA with InternVL on the frames a request names.

POST /vqa   {video_path, ref_indices, question, options, answer, seed}
            -> {answer_log_probability, candidates_log_probabilities}
GET  /health

Requests flow through three pipelined threads (frame decoding -> prompt
building -> model forward), each batching up to max_batch_size requests that
arrive within max_delay seconds of each other.
"""

import argparse
import asyncio
import random
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List

import torch
import torchvision.transforms as T
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

from refocus.utils.videoloader import VideoLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Please choose the correct answer "
    "to the multiple-choice question regarding a video. "
    "The answer must be a single uppercase letter."
)


class VQARequest(BaseModel):
    video_path: str
    ref_indices: List[int]
    question: str
    options: List[str]
    answer: int  # 1-based index into options
    seed: int  # the options are shuffled with this seed


class RequestItem:
    def __init__(self, request: VQARequest):
        self.request = request
        self.timestamp = time.time()
        self.loop = asyncio.get_running_loop()
        self.future = self.loop.create_future()

    # asyncio futures are not thread-safe; worker threads resolve them via the loop.
    def resolve(self, result):
        self.loop.call_soon_threadsafe(self.future.set_result, result)

    def fail(self, exc):
        self.loop.call_soon_threadsafe(self.future.set_exception, exc)


@dataclass
class Sample:
    prompt: str
    answer: int  # token id of the correct option letter
    candidates: List[int]  # token ids of all option letters
    frames: torch.Tensor  # [F, 3, H, W]


@dataclass
class Batch:
    encoding: Dict[str, torch.Tensor]
    answer: torch.Tensor
    candidates: List[torch.Tensor]
    pixel_values: torch.Tensor


class FrameLoader:
    def __init__(self, num_workers):
        self.videoloader = VideoLoader(
            target_size=(INPUT_SIZE, INPUT_SIZE), num_workers=num_workers
        )

    @torch.inference_mode()
    def get_frames_bulk(self, requests):
        """Yields (request position, normalized frames); each video is decoded once."""
        by_video = {}
        for idx, (video_path, indices) in enumerate(requests):
            idx_, indices_, length_ = by_video.setdefault(video_path, ([], [], []))
            idx_.append(idx)
            indices_.extend(indices)
            length_.append(len(indices))
        for video_path, (idx_, indices_, length_) in by_video.items():
            frames = torch.from_numpy(self.videoloader.run(video_path, indices_))
            frames = frames.to(device, dtype).permute(0, 3, 1, 2).div_(255)
            frames = T.functional.normalize(frames, IMAGENET_MEAN, IMAGENET_STD, inplace=True)
            for i, f in zip(idx_, frames.split(length_, dim=0)):
                yield i, f


def build_sample(request: VQARequest, frames: torch.Tensor) -> Sample:
    order = list(range(len(request.options)))
    random.Random(request.seed).shuffle(order)
    options = [request.options[i] for i in order]
    letters = [chr(ord("A") + i) for i in range(len(options))]
    answer = letters[order.index(request.answer - 1)]
    instruction = (
        request.question
        + "\nOptions: "
        + "\n".join(f"{letter}. {opt}" for letter, opt in zip(letters, options))
    )
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "".join(f"Frame{i+1}: <image>\n" for i in range(len(frames)))
            + instruction,
        },
        {"role": "assistant", "content": "Answer:\n"},
    ]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, continue_final_message=True
    )
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token + IMG_END_TOKEN
    prompt = prompt.replace("<image>", image_tokens, len(frames))
    return Sample(prompt, vocab[answer], [vocab[c] for c in letters], frames)


@torch.inference_mode()
def collate(samples: List[Sample]) -> Batch:
    return Batch(
        pixel_values=torch.cat([s.frames for s in samples]),
        # Padding to a multiple of 256 keeps the number of distinct sequence shapes small.
        encoding=tokenizer(
            [s.prompt for s in samples],
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=256,
            padding_side="left",
        ).to(device),
        answer=torch.tensor([s.answer for s in samples], device=device),
        candidates=[torch.tensor(s.candidates, device=device) for s in samples],
    )


@torch.inference_mode()
def probability(batch: Batch):
    log_dist = (
        model(
            **batch.encoding,
            pixel_values=batch.pixel_values,
            image_flags=torch.ones(len(batch.pixel_values), device=device),
            use_cache=False,
            return_dict=True,
        )
        .logits.squeeze(1)
        .log_softmax(dim=-1, dtype=torch.float64)
    )
    answer_logp = log_dist.gather(1, batch.answer[:, None]).squeeze(1).tolist()
    candidates_logp = [
        log_dist[i].gather(0, cand).tolist() for i, cand in enumerate(batch.candidates)
    ]
    return answer_logp, candidates_logp


def drain(buffer, cv, timestamp_of, max_take):
    """Waits until buffer holds max_batch_size items or its oldest item is
    max_delay old, then pops up to max_take of them."""
    with cv:
        while not buffer or (
            len(buffer) < max_batch_size
            and (wait := timestamp_of(buffer[0]) + max_delay - time.time()) > 0
        ):
            cv.wait(timeout=wait if buffer else None)
        items = [buffer.popleft() for _ in range(min(max_take, len(buffer)))]
        cv.notify()
        return items


def put(buffer, cv, item):
    with cv:
        while len(buffer) == buffer.maxlen:
            cv.wait()
        buffer.append(item)
        cv.notify()


def pre_processing_loop():
    while True:
        items = drain(request_buffer, request_cv, lambda it: it.timestamp, 2 * max_batch_size)
        bulk = [(it.request.video_path, it.request.ref_indices) for it in items]
        done = set()
        try:
            for order, frames in frame_loader.get_frames_bulk(bulk):
                item = items[order]
                done.add(order)
                try:
                    put(sample_buffer, sample_cv, (item, build_sample(item.request, frames)))
                except Exception as e:
                    item.fail(e)
        except Exception as e:
            for order, item in enumerate(items):
                if order not in done:
                    item.fail(e)


def batch_processing_loop():
    while True:
        items = drain(sample_buffer, sample_cv, lambda it: it[0].timestamp, max_batch_size)
        put(batch_buffer, batch_cv, (collate([sample for _, sample in items]), items))


def model_processing_loop():
    while True:
        with batch_cv:
            while not batch_buffer:
                batch_cv.wait()
            batch, items = batch_buffer.popleft()
            batch_cv.notify()
        try:
            answer_logp, candidates_logp = probability(batch)
        except Exception as e:
            for item, _ in items:
                item.fail(e)
            continue
        for (item, _), a, c in zip(items, answer_logp, candidates_logp):
            item.resolve({"answer_log_probability": a, "candidates_log_probabilities": c})


class LastTokenOPS(torch.nn.Module):
    """Applies the wrapped head to the last position only."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, inputs: torch.Tensor):
        return self.layer(inputs[:, -1:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_batch_size", type=int, required=True)
    parser.add_argument("--max_delay", type=float, required=True, help="seconds a request waits for batch-mates")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    max_batch_size = args.max_batch_size
    max_delay = args.max_delay
    device = torch.device("cuda")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab = tokenizer.get_vocab()
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=dtype,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device,
    ).eval()
    lm = model.language_model
    lm.set_output_embeddings(LastTokenOPS(lm.get_output_embeddings()))
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    frame_loader = FrameLoader(num_workers=8)

    request_buffer, request_cv = deque(), threading.Condition()
    sample_buffer, sample_cv = deque(maxlen=2 * max_batch_size), threading.Condition()
    batch_buffer, batch_cv = deque(maxlen=2), threading.Condition()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for target in (pre_processing_loop, batch_processing_loop, model_processing_loop):
            threading.Thread(target=target, daemon=True).start()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.post("/vqa")
    async def vqa_endpoint(request: VQARequest):
        item = RequestItem(request)
        with request_cv:
            request_buffer.append(item)
            request_cv.notify()
        return await item.future

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    uvicorn.run(app, host="0.0.0.0", port=args.port)
