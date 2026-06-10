import re
from pathlib import Path

# hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# base_cache_dir = os.path.expanduser(hf_home)
base_cache_dir = Path("./datasets/")


def lvbench_doc_to_visual(doc):
    video_name = doc.get("video_path") or doc.get("video_name")
    if not video_name:
        raise KeyError("LVBench doc must contain either 'video_path' or 'video_name'")

    video_path = Path(video_name)
    candidate_paths = []
    cache_dirs = [base_cache_dir / "LVBench"]
    for cache_dir in cache_dirs:
        candidate_paths.append(cache_dir / video_path)
        candidate_paths.append(cache_dir / "videos" / video_path.name)

    seen = set()
    for candidate_path in candidate_paths:
        if candidate_path in seen:
            continue
        seen.add(candidate_path)
        if candidate_path.exists():
            return [str(candidate_path)]

    checked_paths = "\n".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"video path for {video_name} does not exist, checked:\n{checked_paths}")


def lvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    if "pre_prompt" not in lmms_eval_specific_kwargs:
        lmms_eval_specific_kwargs["pre_prompt"] = ""
    if "post_prompt" not in lmms_eval_specific_kwargs:
        lmms_eval_specific_kwargs["post_prompt"] = "\nAnswer the question with the option letter"
    return lmms_eval_specific_kwargs["pre_prompt"] + doc["question"] + lmms_eval_specific_kwargs["post_prompt"]


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def lvbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")
    gt_ans = doc["answer"]
    score = pred_ans == gt_ans

    # return {f"videomme_perception_score": data_dict for metric in matrices}
    return {"lvbench_score": score}
