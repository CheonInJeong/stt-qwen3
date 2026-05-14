# 프로젝트 요구사항 (Qwen3-ASR 0.6B 기반 콜봇 STT 실험)

## 1. 목표

로컬 CPU 환경에서 다음을 실험할 수 있는 시스템 구축:

> **“콜봇 질문 + 음성 입력을 함께 활용하여 Qwen3-ASR 0.6B 기반 STT 정확도를 개선하는 contextual ASR 실험 환경”**

---

# 2. 핵심 개념

## 기존 STT

```text
audio → text
```

## 개선 STT (핵심 실험)

```text
question + audio → text
```

목적:

- 짧은 발화 (“네”, “맞아요”, “아니요”) 정확도 개선
- 잡음 환경에서 robustness 향상

---

# 3. 모델 요구사항

## ✔ 모델

- Qwen3-ASR 0.6B 사용

## ✔ 실행 환경

- CPU only 지원 필수
- GPU 없어도 동작해야 함
- inference 최적화 필요 (torch CPU mode / quantization 고려)

---

# 4. 시스템 구성

## ✔ 전체 구조

```text
[Web UI]
   ↓
[FastAPI Backend]
   ↓
[Qwen3-ASR 0.6B Model]
   ↓
[STT 결과 출력]
```

---

# 5. 웹 UI 요구사항 (필수)

간단한 테스트 페이지 구현

## 기능

### 1) 질문 입력

- 콜봇 질문 텍스트 입력
- 예: “이 전화번호 맞으십니까?”

---

### 2) 음성 입력

- 마이크 녹음 버튼
- 또는 wav 파일 업로드

---

### 3) 실행 버튼

- STT 실행

---

### 4) 결과 출력

- STT 결과 텍스트

---

### 5) 비교 모드 (중요)

| 모드     | 설명                 |
| -------- | -------------------- |
| baseline | audio only STT       |
| context  | question + audio STT |

👉 결과 비교 UI 반드시 포함

---

# 6. API 설계 (FastAPI)

## ✔ 1. 기본 STT

```http
POST /stt
```

### input

- audio file

### output

- text

---

## ✔ 2. Context-aware STT

```http
POST /stt/context
```

### input

- audio file
- question text

### output

- text

---

# 7. 학습 데이터 구조 (중요)

## ✔ JSONL 형태

```json
{
  "question": "이 전화번호 맞으십니까?",
  "audio_path": "data/001.wav",
  "label": "네"
}
```

---

## ✔ 학습 목적

Qwen3-ASR 0.6B를 fine-tuning 하여:

> audio + question → 정확한 transcription

---

# 8. 학습 요구사항

1. PyTorch 기반

2. HuggingFace 가능

3. fine-tuning 구조 포함

---

## ✔ 모델 입력 구조

```text
input:
  question text
  audio features

output:
  transcription (label)
```

---

# 9. CPU 최적화 요구

필수 조건:

- FP16 없이 CPU inference 가능
- model loading lightweight
- optional:
  - quantization (int8)
  - torch.compile (if possible)

---

# 10. 비교 실험 기능 (중요)

웹에서 반드시 비교 가능해야 함:

## 결과 비교

| 입력        | baseline | context |
| ----------- | -------- | ------- |
| “네”(noise) | 결과 A   | 결과 B  |

👉 성능 차이 확인 목적

---

# 11. 추가 기능 (선택)

- noise injection 테스트
- short utterance dataset 포함
- latency 측정
- confidence score 출력

---

# 12. 핵심 요구

“Qwen3-ASR 0.6B 모델을 CPU 환경에서 실행하고, 콜봇 질문 + 음성을 함께 입력하여 contextual STT 성능을 비교/학습/테스트할 수 있는 웹 기반 실험 시스템을 FastAPI + Python으로 구현”
