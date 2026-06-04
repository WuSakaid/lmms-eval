import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
# TASK_TYPES = ["TR", "AR", "VS", "NQA", "ER", "PQA", "SSC", "AO", "AC"]
TASK_TYPES = ["anomaly_reco", "count", "ego", "needle", "order", "plotQA", "topic_reasoning"]
# hf_home = os.getenv("HF_HOME", "./~/.cache/huggingface")
# base_cache_dir = os.path.expanduser(hf_home)
workspace_root = Path(__file__).resolve().parents[4]
base_cache_dir = workspace_root / "datasets"
with open(Path(__file__).parent / "mlvu.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def mlvu_doc_to_visual(doc):
    video_name = doc.get("video_name")

    video_file = Path(video_name)
    if video_file.is_absolute() and video_file.exists():
        return [str(video_file)]

    cache_dirs = [base_cache_dir / cache_name, base_cache_dir / "MLVU"]
    source_stem = Path(doc.get("source_file", "")).stem
    candidate_paths = []
    for cache_dir in cache_dirs:
        candidate_paths.append(cache_dir / "video" / video_name)
        candidate_paths.append(cache_dir / video_name)
        if source_stem:
            candidate_paths.append(cache_dir / "video" / source_stem / video_name)

    seen = set()
    for candidate_path in candidate_paths:
        if candidate_path in seen:
            continue
        seen.add(candidate_path)
        if candidate_path.exists():
            return [str(candidate_path)]

    checked_paths = "\n".join(str(path) for path in candidate_paths)
    sys.exit(f"video path for {video_name} does not exist, checked:\n{checked_paths}")


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
    match = re.search(r"\(([A-Za-z])\)", s)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-Za-z])\b", s)
    if match:
        return match.group(1).upper()
    return s[:1].upper()



def mlvu_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    # print("****************",pred)
    pred_ans = extract_characters_regex(pred)

    task_type = doc.get("task_type")
    data_dict = {"question_id": doc["question"], "task_type": task_type, "pred_answer": pred_ans, "answer": doc["answer"]}

    return {f"mlvu_percetion_score": data_dict}


def mlvu_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = defaultdict(lambda: {"correct": 0, "answered": 0})
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    task_types = TASK_TYPES + [task_type for task_type in category2score.keys() if task_type not in TASK_TYPES]
    for task_cate in task_types:
        total_correct = category2score[task_cate]["correct"]
        total_answered = category2score[task_cate]["answered"]
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    return 100 * total_correct / total_answered if total_answered > 0 else 0
