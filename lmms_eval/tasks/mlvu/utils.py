import os
import re
from pathlib import Path

import yaml
from loguru import logger as eval_logger

workspace_root = Path(__file__).resolve().parents[4]
base_cache_dir = workspace_root / "datasets"


def _resolve_dataset_dir(task_yaml_name):
    with (Path(__file__).parent / task_yaml_name).open(encoding="utf-8") as handle:
        config = yaml.safe_load("".join(line for line in handle if "!function" not in line))
    dataset_kwargs = config["dataset_kwargs"]
    cache_dir = Path(os.path.expanduser(os.path.expandvars(dataset_kwargs["cache_dir"])))
    if cache_dir.is_absolute():
        return cache_dir
    if "data_files" in dataset_kwargs:
        # AKS local annotations live alongside video/ and its source subfolders.
        return (base_cache_dir if len(cache_dir.parts) == 1 else workspace_root) / cache_dir
    return Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))) / cache_dir


cache_dir_dev = _resolve_dataset_dir("mlvu_dev.yaml")
cache_dir_test = _resolve_dataset_dir("mlvu_test.yaml")


def _mlvu_doc_to_visual(doc, cache_dir):
    video_name = doc.get("video_name")
    if not video_name:
        raise ValueError("MLVU documents require a video_name")
    video_file = Path(os.path.expanduser(os.path.expandvars(video_name)))
    if video_file.is_absolute() and video_file.is_file():
        return [str(video_file)]

    candidates = [cache_dir / "video" / video_file, cache_dir / video_file]
    source_stem = Path(doc.get("source_file") or "").stem
    if source_stem:
        candidates.append(cache_dir / "video" / source_stem / video_file)
    for candidate in candidates:
        if candidate.is_file():
            return [str(candidate)]
    checked_paths = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"MLVU video {video_name} does not exist; checked:\n{checked_paths}")


def mlvu_doc_to_visual_dev(doc):
    return _mlvu_doc_to_visual(doc, cache_dir_dev)


def mlvu_doc_to_visual_test(doc):
    return _mlvu_doc_to_visual(doc, cache_dir_test)


def mlvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()

    option_prompt = (
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
    )

    full_prompt = (
        option_prompt
        + "\n"
        + question
        + "\n"
        + "Your answer must be exactly one character: A, B, C, or D.\n"
        + "Answer:"
    )
    return full_prompt


def extract_characters_regex(s):
    s = s.strip()
    match = re.search(r"\(([A-D])\)", s, re.IGNORECASE)
    if match is None:
        match = re.search(r"\b([A-D])\b", s, re.IGNORECASE)
    return match.group(1).upper() if match else s


def mlvu_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]

    pred_ans = extract_characters_regex(pred)

    task_type = doc["task_type"]
    data_dict = {"question_id": doc["question"], "task_type": task_type, "pred_answer": pred_ans, "answer": doc["answer"]}

    return {"mlvu_percetion_score": data_dict}


def mlvu_aggregate_results_dev(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    TASK_TYPES = {"anomaly_reco", "count", "ego", "needle", "order", "plotQA", "topic_reasoning"}
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    task_category_scores = {}

    # Calculate and log accuracy for each task category
    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")

    # Calculate and log average accuracy across all task categories
    if TASK_TYPES:
        average_accuracy = sum(task_category_scores.values()) / len(TASK_TYPES)
    else:
        average_accuracy = 0

    eval_logger.info(f"Average Performance Across All Task Categories: {average_accuracy:.1f}%")

    return average_accuracy


def mlvu_aggregate_results_test(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    TASK_TYPES = {"anomaly_reco", "count", "ego", "needleQA", "order", "plotQA", "sportsQA", "topic_reasoning", "tutorialQA"}
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    task_category_scores = {}

    # Calculate and log accuracy for each task category
    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")

    # Calculate and log average accuracy across all task categories
    if TASK_TYPES:
        average_accuracy = sum(task_category_scores.values()) / len(TASK_TYPES)
    else:
        average_accuracy = 0

    eval_logger.info(f"Average Performance Across All Task Categories: {average_accuracy:.1f}%")

    return average_accuracy
