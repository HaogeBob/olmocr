import argparse
import json
import os
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from opencc import OpenCC
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

from olmocr.train.config import Config
from olmocr.train.dataloader import BaseMarkdownPDFDataset


def calculate_char_accuracy(pred: str, label: str) -> float:
    """
    基于匹配字符数 + label 长度计算 acc_label。
    acc_label: 匹配字符数 / len(label)
    逻辑与训练评估中的 SequenceMatcher 计算一致。
    """
    pred = pred.replace(" ", "").replace("\n", "").replace("\t", "")
    label = label.replace(" ", "").replace("\n", "").replace("\t", "")

    len_label = len(label)
    if len_label == 0:
        return 1.0 if len(pred) == 0 else 0.0

    matcher = SequenceMatcher(None, pred, label)
    match_cnt = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            match_cnt += (i2 - i1)

    return match_cnt / len_label if len_label > 0 else 0.0


def strip_front_matter(text: str) -> str:
    if text.startswith("---\n"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].lstrip("\n")
    return text


class QwenDataCollator:
    """Data collator for vision-language models that handles numpy arrays."""

    def __init__(self, max_token_len: Optional[int] = None):
        self.max_token_len = max_token_len

    def __call__(self, examples):
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "pixel_values": [],
            "image_grid_thw": [],
            "meta": [],
        }

        for example in examples:
            if example is None:
                continue

            meta = {}
            if "markdown_path" in example:
                meta["markdown_path"] = str(example["markdown_path"])
            if "pdf_path" in example:
                meta["pdf_path"] = str(example["pdf_path"])
            batch["meta"].append(meta)

            input_ids = torch.from_numpy(example["input_ids"]) if isinstance(example["input_ids"], np.ndarray) else example["input_ids"]
            attention_mask = (
                torch.from_numpy(example["attention_mask"]) if isinstance(example["attention_mask"], np.ndarray) else example["attention_mask"]
            )
            labels = torch.from_numpy(example["labels"]) if isinstance(example["labels"], np.ndarray) else example["labels"]

            if self.max_token_len is not None:
                input_ids = input_ids[: self.max_token_len]
                attention_mask = attention_mask[: self.max_token_len]
                labels = labels[: self.max_token_len]

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

            pixel_values = example["pixel_values"]
            if isinstance(pixel_values, np.ndarray):
                pixel_values = torch.from_numpy(pixel_values)
            batch["pixel_values"].append(pixel_values)

            image_grid_thw = example["image_grid_thw"]
            if isinstance(image_grid_thw, np.ndarray):
                image_grid_thw = torch.from_numpy(image_grid_thw)
            batch["image_grid_thw"].append(image_grid_thw)

        if not batch["input_ids"]:
            return None

        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "labels": torch.stack(batch["labels"]),
            "pixel_values": torch.stack(batch["pixel_values"]),
            "image_grid_thw": torch.stack(batch["image_grid_thw"]),
            "meta": batch["meta"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="复现训练评估的推理与 SequenceMatcher 指标计算流程")
    parser.add_argument("--config", type=str, required=True, help="训练配置 YAML 路径")
    parser.add_argument("--dataset-type", choices=["train", "eval"], default="eval", help="使用 train 或 eval 数据集")
    parser.add_argument("--dataset-index", type=int, default=0, help="使用的 dataset 索引")
    parser.add_argument("--max-samples", type=int, default=0, help="最多评估多少条，0 表示全量")
    parser.add_argument("--max-new-tokens", type=int, default=5120, help="生成时最大新 token 数")
    parser.add_argument("--device", type=str, default="cuda", help="指定设备，例如 cuda 或 cpu")
    parser.add_argument("--output-json", type=str, default="sequence_matcher_results.json", help="输出 JSON 文件名")

    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    config.validate()

    processor = AutoProcessor.from_pretrained(config.model.name)
    cc = OpenCC("t2s")

    model_init_kwargs = {
        "torch_dtype": getattr(torch, config.model.torch_dtype) if config.model.torch_dtype != "auto" else "auto",
        "device_map": config.model.device_map,
        "trust_remote_code": config.model.trust_remote_code,
        "attn_implementation": config.model.attn_implementation if config.model.use_flash_attention else None,
    }

    if "qwen2.5-vl" in config.model.name.lower() or "olmocr" in config.model.name.lower():
        model_class = Qwen2_5_VLForConditionalGeneration
    elif "qwen2-vl" in config.model.name.lower():
        model_class = Qwen2VLForConditionalGeneration
    else:
        raise NotImplementedError("仅支持 Qwen2-VL / Qwen2.5-VL 模型")

    model = model_class.from_pretrained(config.model.name, **model_init_kwargs)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.model.device_map is None:
        model.to(device)

    dataset_cfgs = config.dataset.train if args.dataset_type == "train" else config.dataset.eval
    if args.dataset_index >= len(dataset_cfgs):
        raise SystemExit(f"dataset-index 超出范围：{args.dataset_index}")

    dataset_cfg = dataset_cfgs[args.dataset_index]
    root_dir = dataset_cfg["root_dir"]
    pipeline_steps = config.get_pipeline_steps(dataset_cfg["pipeline"], processor)
    dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps)

    batch_size = config.training.per_device_eval_batch_size if args.dataset_type == "eval" else config.training.per_device_train_batch_size
    data_collator = QwenDataCollator(max_token_len=config.training.collator_max_token_len)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=config.training.dataloader_num_workers,
        drop_last=False,
    )

    model.eval()
    total_char_acc = 0.0
    num_gen_samples = 0
    results = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

            input_ids = batch["input_ids"]
            labels = batch["labels"]
            metas = batch.get("meta", [])

            for i in range(len(input_ids)):
                lbl = labels[i]
                valid_indices = (lbl != -100).nonzero(as_tuple=True)[0]
                if len(valid_indices) == 0:
                    continue

                start_idx = valid_indices[0].item()
                prompt_ids = input_ids[i : i + 1, :start_idx]
                prompt_mask = batch["attention_mask"][i : i + 1, :start_idx]

                gen_kwargs = {
                    "input_ids": prompt_ids,
                    "attention_mask": prompt_mask,
                    "max_new_tokens": args.max_new_tokens,
                    "use_cache": True,
                }
                if "pixel_values" in batch:
                    gen_kwargs["pixel_values"] = batch["pixel_values"][i : i + 1]
                if "image_grid_thw" in batch:
                    gen_kwargs["image_grid_thw"] = batch["image_grid_thw"][i : i + 1]

                with autocast(device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                    generated_ids = model.generate(**gen_kwargs)

                new_tokens = generated_ids[0, start_idx:]
                pred_text = processor.decode(new_tokens, skip_special_tokens=True)

                target_ids = lbl[lbl != -100]
                target_text = processor.decode(target_ids, skip_special_tokens=True)

                pred_text = strip_front_matter(pred_text)
                target_text = strip_front_matter(target_text)

                pred_text = cc.convert(pred_text)
                target_text = cc.convert(target_text)

                char_acc = calculate_char_accuracy(pred_text, target_text)
                total_char_acc += char_acc
                num_gen_samples += 1

                meta = metas[i] if i < len(metas) else {}
                results.append(
                    {
                        "index": num_gen_samples - 1,
                        "markdown_path": meta.get("markdown_path"),
                        "pdf_path": meta.get("pdf_path"),
                        "label": target_text,
                        "pred": pred_text,
                        "score": char_acc,
                    }
                )

                if args.max_samples > 0 and num_gen_samples >= args.max_samples:
                    break

            if args.max_samples > 0 and num_gen_samples >= args.max_samples:
                break

    avg_char_acc = total_char_acc / num_gen_samples if num_gen_samples > 0 else 0.0
    print(f"char_acc: {avg_char_acc:.6f}")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
