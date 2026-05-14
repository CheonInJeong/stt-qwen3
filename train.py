"""
Qwen3-ASR 0.6B 파인튜닝 스크립트

[파인튜닝이란?]
이미 대용량 데이터로 학습된 모델을 우리 도메인(콜봇 한국어)에 맞게
추가로 학습시키는 것입니다. 처음부터 학습하는 것보다 훨씬 적은
데이터와 시간으로 정확도를 높일 수 있습니다.

[학습 데이터 형식 - JSONL]
한 줄에 JSON 한 개씩:
  {"id": "abc123", "question": "성함이 어떻게 되세요?", "audio_path": "data/audio/abc123.wav", "label": "홍길동"}

사용법:
  python train.py --data data/dataset.jsonl --output ./finetuned --epochs 3
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor, get_linear_schedule_with_warmup

# qwen_asr를 임포트하면 Qwen3ASR 모델 클래스가 HuggingFace AutoModel에 등록됩니다.
# 이 한 줄이 없으면 AutoModel.from_pretrained()가 모델 타입을 인식하지 못합니다.
import qwen_asr  # noqa: F401 — 등록 목적의 임포트

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # 서버가 stdout을 읽기 때문
)
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
SAMPLE_RATE = 16_000  # 16kHz — Qwen3-ASR이 요구하는 샘플링 레이트


# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋 클래스
# ─────────────────────────────────────────────────────────────────────────────

class CallbotDataset(Dataset):
    """
    JSONL 파일에서 샘플을 읽어 학습용 텐서로 변환합니다.

    [SFT(Supervised Fine-Tuning)란?]
    모델에게 "이 오디오를 들으면 이 텍스트를 출력해라" 라고 정답을 알려주며 학습시킵니다.
    - 입력(Input)  : 오디오 + 질문(context)
    - 정답(Label)  : 올바른 전사 텍스트
    - 손실(Loss)   : 모델이 예측한 텍스트 vs 정답 텍스트의 차이
    이 차이(손실)를 줄이는 방향으로 모델의 가중치(숫자들)를 조금씩 수정합니다.
    """

    def __init__(self, jsonl_path: str, processor):
        self.processor = processor
        self.samples: List[dict] = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # 오디오 파일이 실제로 존재하는 샘플만 사용합니다.
                if Path(obj["audio_path"]).exists():
                    self.samples.append(obj)
                else:
                    logger.warning(f"오디오 파일 없음, 건너뜀: {obj['audio_path']}")

        logger.info(f"{len(self.samples)}개 샘플 로드 완료 ({jsonl_path})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[dict]:
        s = self.samples[idx]
        question = s.get("question", "")
        label    = s["label"]

        # ── 1. 오디오 로드 ──────────────────────────────────────────────────
        try:
            audio, _ = librosa.load(s["audio_path"], sr=SAMPLE_RATE, mono=True)
            audio = audio.astype(np.float32)
        except Exception as e:
            logger.warning(f"오디오 로드 실패: {s['audio_path']} — {e}")
            return None

        # ── 2. 프롬프트(Prompt) 구성 ─────────────────────────────────────────
        # 모델에게 보여줄 입력 텍스트입니다.
        # Qwen3-ASR은 채팅 템플릿 형식을 사용합니다:
        #   system: 질문(context)
        #   user:   <오디오 플레이스홀더>
        #   assistant: language Korean<asr_text>  ← 모델이 이 다음을 생성해야 함
        messages = [
            {"role": "system", "content": question or ""},
            {"role": "user",   "content": [{"type": "audio", "audio": ""}]},
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # 언어를 Korean으로 고정하는 접두어를 추가합니다 (추론과 동일한 방식)
        prompt += "language Korean<asr_text>"

        # ── 3. 정답(Target) 구성 ─────────────────────────────────────────────
        # 모델이 생성해야 하는 텍스트입니다.
        # <|im_end|>는 "응답 끝" 토큰입니다.
        eos = self.processor.tokenizer.eos_token or "<|im_end|>"
        target = f"{label}</asr_text>{eos}"
        full_text = prompt + target

        # ── 4. 프로세서로 텍스트 + 오디오 처리 ──────────────────────────────
        # 텍스트는 토큰 ID로, 오디오는 멜 스펙트로그램(숫자 행렬)으로 변환됩니다.
        try:
            inputs = self.processor(
                text=[full_text],
                audio=[audio],
                return_tensors="pt",
            )
        except Exception as e:
            logger.warning(f"프로세서 처리 실패: {e}")
            return None

        # ── 5. 레이블(Labels) 생성 ───────────────────────────────────────────
        # [핵심 개념] 학습 시 모델은 "다음 토큰 예측"을 합니다.
        # 하지만 프롬프트 부분은 예측 대상이 아닙니다(정답이 없는 부분).
        # -100으로 표시된 위치는 손실 계산에서 제외됩니다.
        #
        # 예시:
        #   input_ids:  [시스템 토큰들...][오디오 플레이스홀더들...][assistant...][정답 토큰들]
        #   labels:     [  -100 -100 ...][     -100 -100 ...      ][ -100 ...  ][실제 ID들  ]
        #
        # 이렇게 하면 모델은 정답 텍스트 부분만 학습합니다.

        # 정답 텍스트의 토큰 개수를 셉니다
        target_ids = self.processor.tokenizer(
            target, return_tensors="pt", add_special_tokens=False
        ).input_ids
        target_len = target_ids.shape[1]

        # 전체 레이블을 -100으로 초기화한 뒤, 마지막 target_len개만 실제 값으로 채웁니다
        labels = torch.full_like(inputs["input_ids"], -100)
        if target_len > 0 and target_len <= inputs["input_ids"].shape[1]:
            labels[:, -target_len:] = inputs["input_ids"][:, -target_len:]

        return {
            "input_ids":             inputs["input_ids"].squeeze(0),
            "attention_mask":        inputs["attention_mask"].squeeze(0),
            "input_features":        inputs["input_features"].squeeze(0),
            "feature_attention_mask": inputs["feature_attention_mask"].squeeze(0),
            "labels":                labels.squeeze(0),
        }


def collate_skip_none(batch):
    """None(로드 실패) 샘플을 제거하는 커스텀 배치 함수."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    # batch_size=1이므로 패딩 없이 그대로 반환합니다
    return batch[0]


# ─────────────────────────────────────────────────────────────────────────────
# 학습 함수
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    """
    [학습 흐름 요약]
    1. 모델 & 프로세서 로드
    2. 데이터셋 로드
    3. 에폭(epoch)마다 모든 샘플을 한 번씩 학습
       - 각 샘플에 대해 손실(loss) 계산
       - 역전파(backpropagation)로 가중치 업데이트
    4. 학습된 모델을 디스크에 저장
    """
    device = torch.device("cpu")
    logger.info(f"학습 장치: {device}")
    logger.info(f"에폭: {args.epochs}  학습률: {args.lr}")

    # ── 모델 & 프로세서 로드 ─────────────────────────────────────────────────
    logger.info(f"모델 로드 중: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, fix_mistral_regex=True)
    base_model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    # model.thinker가 실제 학습 가능한 신경망입니다.
    # Qwen3ASRForConditionalGeneration은 generate()만 있고,
    # 학습용 forward()는 내부 thinker에 있습니다.
    model = base_model.thinker
    model.to(device)
    model.train()  # 학습 모드 활성화 (드롭아웃 등이 켜짐)

    # ── 데이터셋 & DataLoader ────────────────────────────────────────────────
    dataset = CallbotDataset(args.data, processor)
    if len(dataset) == 0:
        logger.error("학습 데이터가 없습니다. UI에서 데이터를 먼저 수집하세요.")
        return

    # DataLoader: 데이터셋에서 샘플을 하나씩 꺼내줍니다.
    # batch_size=1: 오디오 길이가 달라서 배치를 만들기 복잡하므로 한 번에 1개씩 처리합니다.
    # shuffle=True: 매 에폭마다 순서를 섞어 학습 편향을 줄입니다.
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_skip_none,
    )

    # ── 옵티마이저 & 스케줄러 ────────────────────────────────────────────────
    # 옵티마이저(optimizer): 손실을 줄이는 방향으로 가중치를 업데이트하는 알고리즘
    # AdamW는 가장 널리 쓰이는 딥러닝 옵티마이저입니다.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )

    total_steps = len(loader) * args.epochs

    # 학습률 스케줄러: 처음엔 천천히 올리고(warmup), 이후 점점 낮춥니다.
    # 갑자기 큰 학습률로 시작하면 모델이 망가질 수 있기 때문입니다.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    # ── 학습 루프 ────────────────────────────────────────────────────────────
    logger.info(f"학습 시작: {len(dataset)}개 샘플, {args.epochs}에폭")

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        valid_steps = 0

        for step, batch in enumerate(loader, 1):
            if batch is None:
                continue

            # 텐서를 학습 장치(CPU)로 이동합니다
            input_ids              = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask         = batch["attention_mask"].unsqueeze(0).to(device)
            input_features         = batch["input_features"].unsqueeze(0).to(device)
            feature_attention_mask = batch["feature_attention_mask"].unsqueeze(0).to(device)
            labels                 = batch["labels"].unsqueeze(0).to(device)

            # 순전파(forward pass): 모델에 입력을 넣고 손실을 계산합니다
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            if loss is None or torch.isnan(loss):
                logger.warning(f"스텝 {step}: 손실 계산 불가, 건너뜀")
                continue

            # 역전파(backward pass): 손실에 대한 각 가중치의 기울기를 계산합니다
            optimizer.zero_grad()  # 이전 기울기 초기화
            loss.backward()        # 기울기 계산

            # 그래디언트 클리핑: 기울기가 너무 크면 학습이 불안정해지므로 제한합니다
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()   # 가중치 업데이트
            scheduler.step()   # 학습률 업데이트

            epoch_loss += loss.item()
            valid_steps += 1

            if step % 5 == 0 or step == len(loader):
                logger.info(
                    f"에폭 {epoch}/{args.epochs}  "
                    f"스텝 {step}/{len(loader)}  "
                    f"손실={loss.item():.4f}"
                )

        avg = epoch_loss / valid_steps if valid_steps > 0 else float("nan")
        logger.info(f"[에폭 {epoch} 완료] 평균 손실: {avg:.4f}")

    # ── 모델 저장 ────────────────────────────────────────────────────────────
    # 학습 실행마다 타임스탬프 하위 폴더를 만들어 저장합니다.
    # 예: finetuned/run_20240514_153022/
    # → 여러 번 학습해도 이전 결과가 덮어씌워지지 않습니다.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 전체 모델(base_model)을 저장합니다.
    # thinker만 저장하면 나중에 다시 로드할 때 재조립이 필요해서 복잡해집니다.
    base_model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))

    logger.info(f"모델 저장 완료: {out_dir}")
    # UI가 저장 경로를 파싱할 수 있도록 특별한 형식으로 출력합니다.
    print(f"SAVED_MODEL_PATH={out_dir}", flush=True)
    logger.info("학습이 끝났습니다!")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 인수 파싱
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-ASR 파인튜닝")
    p.add_argument("--data",    default="data/dataset.jsonl", help="JSONL 데이터 경로")
    p.add_argument("--output",  default="./finetuned",         help="모델 저장 루트 경로 (하위에 run_타임스탬프 폴더 생성)")
    p.add_argument("--epochs",  type=int,   default=3,        help="학습 에폭 수")
    p.add_argument("--lr",      type=float, default=1e-5,     help="학습률 (Learning Rate)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
