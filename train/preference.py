"""HTTP client for the reward-API servers (train/reward_api/InternVL2_5.py)."""

from concurrent.futures import ThreadPoolExecutor

import requests
from scipy.special import logsumexp

_executor = ThreadPoolExecutor(128)


def _post(params):
    params = dict(params)
    url = params.pop("url")
    try:
        response = requests.post(url, json=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # A failed request becomes an unsuccessful candidate instead of aborting the step.
        print(f"[ERROR] reward request to {url} failed: {e}")
        return None


def _record(response):
    if response is None:
        return {"success": False, "is_correct": False}
    ans_logp = response["answer_log_probability"]
    all_logp = response["candidates_log_probabilities"]
    return {
        "success": bool(-1.0 < logsumexp(all_logp) < 0.01),
        "is_correct": ans_logp == max(all_logp),
        "ans_logp": ans_logp,
        "all_logp": all_logp,
    }


def query_reward_api(params_list):
    """Scores each request (url, video_path, ref_indices, question, options, answer, seed) concurrently."""
    return [_record(r) for r in _executor.map(_post, params_list)]
