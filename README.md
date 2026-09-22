<div align="center">

# ReFoCUS

### Reinforcement-guided Frame Optimization for Contextual Understanding

**IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, 2026**

[\[📜 Paper\]](https://arxiv.org/abs/2506.01274)
[\[🌐 Project Page\]](https://interlive-team.github.io/ReFoCUS)
[\[🤗 Models\]](https://huggingface.co/interlive)

Hosu Lee<sup>1*</sup>, Junho Kim<sup>2*</sup>, Hyunjun Kim<sup>1</sup>, Yong Man Ro<sup>1†</sup>

<sup>1</sup>KAIST · <sup>2</sup>UIUC

</div>

## Introduction

**ReFoCUS** (Reinforcement-guided Frame Optimization for Contextual UnderStanding) is the
first framework to integrate online policy-gradient reinforcement learning into frame-level
optimization for video-LLMs. ReFoCUS learns a frame-selection policy from reward signals
derived from a reference model, capturing the frame combinations that best support
temporally grounded responses. To explore the large combinatorial frame space efficiently,
it uses an autoregressive, query-conditional selection architecture that preserves
contextual consistency while reducing complexity. The policy is learned without explicit
frame-level supervision, and it consistently improves reasoning accuracy across multiple
video QA benchmarks.

## TODO

- [x] Paper release
- [x] [Project page](https://interlive-team.github.io/ReFoCUS)
- [x] Model weights release ([ReFoCUS-1.3B](https://huggingface.co/interlive/ReFoCUS-1.3B))
- [x] Evaluation code
- [x] Training code

## Training

ReFoCUS optimizes [`interlive/video-ma2mba-1.3b`](https://huggingface.co/interlive/video-ma2mba-1.3b)
as a frame-selection policy through frame-preference optimization against an InternVL3-2B
reward model. To reproduce the paper, follow the steps below.

1. Data. Request access to both [interlive/ReFoCUS-393K](https://huggingface.co/datasets/interlive/ReFoCUS-393K)
   and [interlive/ReFoCUS-393K-metadata](https://huggingface.co/datasets/interlive/ReFoCUS-393K-metadata),
   then download them (set `/path/to/meta` and `/path/to/videos` as appropriate).

   ```bash
   uv run hf download interlive/ReFoCUS-393K-metadata --repo-type dataset --local-dir /path/to/meta
   uv run scripts/download_videos.py --video_root /path/to/videos
   uv run scripts/prepare_train_data.py --meta_dir /path/to/meta --video_root /path/to/videos --output_dir /path/to/data
   ```

2. Reward servers. Launch the servers that score the policy's selections. Set `REWARD_MODEL`
   to change the reward model; InternVL2.5, InternVL3, and InternVL3.5 are supported.

   ```bash
   REWARD_MODEL=OpenGVLab/InternVL3-2B uv run bash scripts/launch_reward_api.sh
   ```

3. Training. Each stage `k` of the search-space scaling curriculum (§3.2 of the paper)
   resumes from stage `k-1`.

   ```bash
   export DATA_ROOT=/path/to/data
   export VIDEO_FOLDER=/path/to/videos
   export REMOTE_EVAL_URLS="$(cat work/reward_api/urls.txt)"
   # export REPORT_TO=wandb WANDB_PROJECT=refocus                 # optional: wandb logging
   for STAGE in 1 2 3 4; do
     # export HUB_MODEL_ID=<user>/refocus-1.3b-stage$STAGE        # optional: push to the Hugging Face Hub
     STAGE=$STAGE uv run bash scripts/train_refocus.sh || break
   done
   xargs -I{} kill -9 -- -{} < work/reward_api/pids.txt
   ```

The final checkpoint (`work/train/refocus-1.3b-stage4`) is a drop-in policy for the evaluation
scripts below.

**Hardware.** Training runs as-is on accelerators with 96 GB or more of memory.

## Evaluation

ReFoCUS is evaluated through [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval),
following the steps below.

1. Frame selection. Run the trained ReFoCUS selector on a benchmark. The third argument is
   a run name that labels the selection (replace `<run_name>` with your own).

   ```bash
   uv run bash scripts/prepare.sh interlive/ReFoCUS-1.3b videomme <run_name>
   ```

2. VLM evaluation. Repeat the same selector, benchmark, and run name, then the downstream
   video-LLM. The frame-index database built in step 1 is located automatically.

   ```bash
   uv run bash scripts/evaluate.sh interlive/ReFoCUS-1.3b videomme <run_name> llava-hf/llava-onevision-qwen2-7b-ov-hf
   ```

**Supported benchmarks**

| Benchmark | Task code |
| --- | --- |
| Video-MME | `videomme` |
| LongVideoBench | `longvideobench_val_v` |
| MLVU | `mlvu_dev` |
| NExT-QA (open-ended) | `nextqa_oe_val` / `nextqa_oe_test` |
| ActivityNet-QA | `activitynetqa` |
| VideoChat-GPT | `videochatgpt` |

**Supported downstream VLMs**

| Model series | Example checkpoints |
| --- | --- |
| LLaVA-OneVision | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`<br>`llava-hf/llava-onevision-qwen2-7b-ov-hf` |
| InternVL3 | `OpenGVLab/InternVL3-1B`<br>`OpenGVLab/InternVL3-2B`<br>`OpenGVLab/InternVL3-8B` |
| InternVL3.5 | `OpenGVLab/InternVL3_5-2B`<br>`OpenGVLab/InternVL3_5-4B`<br>`OpenGVLab/InternVL3_5-8B` |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-3B-Instruct`<br>`Qwen/Qwen2.5-VL-7B-Instruct`<br>`Qwen/Qwen2.5-VL-32B-Instruct` |
| Qwen3-VL | `Qwen/Qwen3-VL-4B-Instruct`<br>`Qwen/Qwen3-VL-8B-Instruct`<br>`Qwen/Qwen3-VL-32B-Instruct` |
| VideoLLaMA3 | `DAMO-NLP-SG/VideoLLaMA3-2B`<br>`DAMO-NLP-SG/VideoLLaMA3-7B` |

Other sizes in each series work too. All evaluations require `OPENAI_API_KEY` and `HF_HOME` set in the environment.

## Citation

```bibtex
@InProceedings{Lee_2026_CVPR,
    author    = {Lee, Hosu and Kim, Junho and Kim, Hyunjun and Ro, Yong Man},
    title     = {ReFoCUS: Reinforcement-guided Frame Optimization for Contextual Understanding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {8291-8302}
}
```

## License

This project is released under the [Apache 2.0 License](LICENSE).
