"""
Fine-tuning script for Qwen3-ASR 0.6B
Dataset format (JSONL):
  {"question": "...", "audio_path": "data/001.wav", "label": "네"}

Usage:
  python train.py --data data/sample.jsonl --output ./finetuned --epochs 3
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
SAMPLE_RATE = 16000


# ── Dataset ───────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    question: Optional[str]
    audio_path: str
    label: str


class CallbotSTTDataset(Dataset):
    """Loads JSONL samples; optionally filters to question-only or audio-only."""

    def __init__(self, jsonl_path: str, processor, use_question: bool = True):
        self.samples: List[Sample] = []
        self.processor = processor
        self.use_question = use_question

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(
                    Sample(
                        question=obj.get("question"),
                        audio_path=obj["audio_path"],
                        label=obj["label"],
                    )
                )
        logger.info(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        audio, _ = librosa.load(sample.audio_path, sr=SAMPLE_RATE, mono=True)

        # Audio features
        inputs = self.processor(
            audio.astype(np.float32),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].squeeze(0)  # (80, T)

        # Label token IDs
        label_ids = self.processor.tokenizer(
            sample.label,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.squeeze(0)

        result = {
            "input_features": input_features,
            "labels": label_ids,
        }

        # Optional question context
        if self.use_question and sample.question:
            question_ids = self.processor.tokenizer(
                sample.question,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.squeeze(0)
            result["question_ids"] = question_ids

        return result


def collate_fn(batch: List[dict]) -> dict:
    """Pad sequences and build batch tensors."""
    input_features = torch.stack([b["input_features"] for b in batch])

    # Pad labels to same length
    max_label_len = max(b["labels"].shape[0] for b in batch)
    padded_labels = torch.full((len(batch), max_label_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        llen = b["labels"].shape[0]
        padded_labels[i, :llen] = b["labels"]

    result = {"input_features": input_features, "labels": padded_labels}

    # Pack question_ids as list (variable length — not padded for now)
    if "question_ids" in batch[0]:
        result["question_ids"] = [b.get("question_ids") for b in batch]

    return result


# ── Trainer ───────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cpu")
    logger.info(f"Training on {device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.train()

    dataset = CallbotSTTDataset(
        args.data, processor, use_question=not args.baseline_only
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for step, batch in enumerate(loader, 1):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_features=input_features,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            if step % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{args.epochs}  step {step}/{len(loader)}  "
                    f"loss={loss.item():.4f}"
                )

        avg = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch} average loss: {avg:.4f}")

    # Save fine-tuned model
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    logger.info(f"Model saved to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen3-ASR 0.6B")
    p.add_argument("--data", default="data/sample.jsonl", help="JSONL dataset path")
    p.add_argument("--output", default="./finetuned", help="Output directory")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument(
        "--baseline-only",
        action="store_true",
        dest="baseline_only",
        help="Train without question context (baseline mode)",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
