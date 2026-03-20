#!/usr/bin/env python3
"""
依据 scene->condition 索引文件，对其它结果文件进行分割。

规则：
1) 天气三类：overcast / sunny / rain
2) 光照两类：daytime / night
   - daytime 仅当 condition == "sunny daytime"
   - night 只要 condition 中包含 "night"
3) 天气与光照是两套并行标签，可重复归属。

支持输入文件：.json / .pkl
- 若输入是 list[dict]，按记录分割
- 自动从记录中识别 sample_token 或 scene_token（优先 sample_token）

示例：
python tools/split_by_nuscenes_condition.py \
  --index-json /mnt/vol1/zhaozc_workspace/project/InternVL25/datacopy/nuscenes_scene_sample_index_val.json \
  --input-files /path/file1.json,/path/file2.pkl \
  --output-dir /mnt/vol1/zhaozc_workspace/project/InternVL25/datacopy/splits
"""

import argparse
import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Set, Tuple

HEX32_RE = re.compile(r"^[0-9a-f]{32}$")


def normalize_condition(x: str): #-> str:
    return (x or "").strip().lower()


def load_index(index_json: str): #-> Tuple[Dict[str, str], Dict[str, str], Set[str], Set[str]]:
    """
    返回：
    - scene_to_condition
    - sample_to_condition
    - all_scene_tokens
    - all_sample_tokens
    """
    with open(index_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = data.get("scenes", []) if isinstance(data, dict) else []
    scene_to_condition: Dict[str, str] = {}
    sample_to_condition: Dict[str, str] = {}

    for s in scenes:
        if not isinstance(s, dict):
            continue
        scene_token = str(s.get("scene_token", "")).strip()
        condition = normalize_condition(str(s.get("condition", "")))
        if not scene_token:
            continue

        scene_to_condition[scene_token] = condition

        # 优先使用全量 sample_tokens；如果没有则退化为 selected_sample_tokens
        sample_tokens = s.get("sample_tokens", [])
        if not sample_tokens:
            sample_tokens = s.get("selected_sample_tokens", [])

        for tok in sample_tokens:
            tok = str(tok).strip()
            if tok:
                sample_to_condition[tok] = condition

    return scene_to_condition, sample_to_condition, set(scene_to_condition.keys()), set(sample_to_condition.keys())


def is_hex32(s: str): #-> bool:
    return bool(HEX32_RE.match(s))


def collect_possible_tokens(obj: Any, out: Optional[Set[str]] = None): #-> Set[str]:
    if out is None:
        out = set()

    if isinstance(obj, dict):
        for k, v in obj.items():
            k_low = str(k).lower()
            if isinstance(v, str) and ("token" in k_low):
                vv = v.strip().lower()
                if is_hex32(vv):
                    out.add(vv)
            collect_possible_tokens(v, out)
    elif isinstance(obj, list):
        for x in obj:
            collect_possible_tokens(x, out)
    elif isinstance(obj, str):
        vv = obj.strip().lower()
        if is_hex32(vv):
            out.add(vv)

    return out


def infer_condition_for_record(
    record: Dict[str, Any],
    scene_to_condition: Dict[str, str],
    sample_to_condition: Dict[str, str],
    all_scene_tokens: Set[str],
    all_sample_tokens: Set[str],
): #-> str:
    # 1) 快速路径：常见字段
    for key in ["sample_token", "token", "sampleToken"]:
        val = record.get(key)
        if isinstance(val, str):
            tok = val.strip().lower()
            if tok in sample_to_condition:
                return sample_to_condition[tok]

    for key in ["scene_token", "sceneToken"]:
        val = record.get(key)
        if isinstance(val, str):
            tok = val.strip().lower()
            if tok in scene_to_condition:
                return scene_to_condition[tok]

    # 2) 深度搜索 token
    candidates = collect_possible_tokens(record)

    # 优先 sample token
    for tok in candidates:
        if tok in all_sample_tokens:
            return sample_to_condition.get(tok, "")

    # 再看 scene token
    for tok in candidates:
        if tok in all_scene_tokens:
            return scene_to_condition.get(tok, "")

    return ""


def load_any(path: str): #-> Any:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported file type: {path}")


def dump_any(path: str, data: Any): #-> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return
    if path.endswith(".pkl"):
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return
    raise ValueError(f"Unsupported file type: {path}")


def split_records(
    records: List[Dict[str, Any]],
    scene_to_condition: Dict[str, str],
    sample_to_condition: Dict[str, str],
    all_scene_tokens: Set[str],
    all_sample_tokens: Set[str],
): #-> Dict[str, List[Dict[str, Any]]]:
    out = {
        "weather_overcast": [],
        "weather_sunny": [],
        "weather_rain": [],
        "lighting_daytime": [],
        "lighting_night": [],
        "unmatched": [],
    }

    for rec in records:
        condition = infer_condition_for_record(
            rec, scene_to_condition, sample_to_condition, all_scene_tokens, all_sample_tokens
        )
        c = normalize_condition(condition)

        hit = False

        # 天气三类
        if "overcast" in c:
            out["weather_overcast"].append(rec)
            hit = True
        if "sunny" in c:
            out["weather_sunny"].append(rec)
            hit = True
        if "rain" in c:
            out["weather_rain"].append(rec)
            hit = True

        # 光照两类
        if c == "sunny daytime":
            out["lighting_daytime"].append(rec)
            hit = True
        if "night" in c:
            out["lighting_night"].append(rec)
            hit = True

        if not hit:
            out["unmatched"].append(rec)

    return out


def process_one_file(
    input_path: str,
    output_dir: str,
    scene_to_condition: Dict[str, str],
    sample_to_condition: Dict[str, str],
    all_scene_tokens: Set[str],
    all_sample_tokens: Set[str],
): #-> None:
    obj = load_any(input_path)
    if not isinstance(obj, list):
        raise TypeError(f"Input must be a list of records: {input_path}")

    records: List[Dict[str, Any]] = [x for x in obj if isinstance(x, dict)]
    split = split_records(records, scene_to_condition, sample_to_condition, all_scene_tokens, all_sample_tokens)

    base = os.path.basename(input_path)
    stem, ext = os.path.splitext(base)

    for key, items in split.items():
        out_path = os.path.join(output_dir, f"{stem}.{key}{ext}")
        dump_any(out_path, items)
        print(f"[{base}] {key}: {len(items)}: #-> {out_path}")


def parse_args(): #-> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-json", type=str, required=True, help="包含 scene_token/condition 的索引 JSON")
    parser.add_argument("--input-files", type=str, required=True, help="逗号分隔的待分割文件列表（json/pkl）")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    return parser.parse_args()


def main(): #-> None:
    args = parse_args()
    scene_to_condition, sample_to_condition, all_scene_tokens, all_sample_tokens = load_index(args.index_json)

    input_files = [x.strip() for x in args.input_files.split(",") if x.strip()]
    if len(input_files) < 1:
        raise ValueError("No input files provided")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(f"index scenes: {len(scene_to_condition)}")
    print(f"index samples: {len(sample_to_condition)}")
    print(f"input files: {len(input_files)}")
    print("=" * 80)

    for p in input_files:
        process_one_file(
            input_path=p,
            output_dir=args.output_dir,
            scene_to_condition=scene_to_condition,
            sample_to_condition=sample_to_condition,
            all_scene_tokens=all_scene_tokens,
            all_sample_tokens=all_sample_tokens,
        )


if __name__ == "__main__":
    main()
