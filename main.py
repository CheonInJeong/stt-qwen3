"""
[이 파일의 역할]
FastAPI 웹 서버입니다. 세 가지 기능을 제공합니다:

1. STT 추론 API  — 음성을 텍스트로 변환
2. 데이터 수집 API — 학습 데이터(음성 + 정답 텍스트)를 저장
3. 학습 API       — 수집한 데이터로 모델을 파인튜닝

[제공하는 엔드포인트]
GET  /               — UI 페이지 (index.html)
GET  /health         — 서버 상태 확인

POST /stt            — 기본 STT (음성만)
POST /stt/context    — 문맥 인식 STT (질문 + 음성)
POST /stt/compare    — 두 모드 동시 비교

POST /data/collect   — 학습 데이터 1건 저장 (음성 + 질문 + 정답)
GET  /data/list      — 수집된 데이터 목록 조회
DELETE /data/{id}    — 특정 데이터 삭제

POST /train/start    — 파인튜닝 시작 (백그라운드에서 train.py 실행)
GET  /train/status   — 학습 진행 상태 & 로그 조회
"""

import json
import logging
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from model import get_model, set_model, get_current_model_id

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 저장 경로
# ─────────────────────────────────────────────────────────────────────────────

# JSONL 파일: 수집한 샘플의 메타데이터(질문, 정답, 오디오 경로)를 저장합니다.
# JSONL = JSON Lines — 한 줄에 JSON 객체 하나씩 저장하는 형식
DATASET_JSONL = Path("data/dataset.jsonl")
AUDIO_DIR     = Path("data/audio")
FINETUNED_DIR = Path("finetuned")      # 파인튜닝 결과가 run_타임스탬프 형태로 저장됩니다
BASE_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"  # 허깅페이스 기본 모델 ID


# ─────────────────────────────────────────────────────────────────────────────
# 학습 프로세스 전역 상태
# ─────────────────────────────────────────────────────────────────────────────
# 학습은 별도 프로세스(train.py)에서 실행됩니다.
# 서버는 이 변수들로 학습 상태를 추적합니다.

_train_proc: Optional[subprocess.Popen] = None  # 실행 중인 학습 프로세스
_train_logs: List[str] = []                      # 학습 로그 (최근 500줄)
_train_lock = threading.Lock()                   # 스레드 간 충돌 방지용 잠금


def _read_proc_output(proc: subprocess.Popen):
    """
    백그라운드 스레드에서 학습 프로세스의 출력을 읽어 _train_logs에 저장합니다.

    [스레드(Thread)란?]
    프로그램 안에서 동시에 실행되는 별도의 실행 흐름입니다.
    학습 로그를 읽는 동안 서버가 다른 요청도 처리할 수 있습니다.
    """
    for line in proc.stdout:
        with _train_lock:
            _train_logs.append(line.rstrip())
            if len(_train_logs) > 500:  # 최대 500줄만 유지
                _train_logs.pop(0)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 앱 생성
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Qwen3-ASR Contextual STT",
    description="콜봇 STT 정확도 실험: 기본 vs 문맥 인식 + 데이터 수집 & 학습",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# 응답 스키마 정의
# ─────────────────────────────────────────────────────────────────────────────

class STTResponse(BaseModel):
    """STT 결과 하나를 담는 데이터 구조."""
    text: str          # 변환된 텍스트
    latency_sec: float # 처리 시간(초)
    mode: str          # "baseline" 또는 "context"


class CompareResponse(BaseModel):
    """비교 모드의 응답: baseline + context 결과를 동시에 반환."""
    baseline: STTResponse
    context: STTResponse


class CollectResponse(BaseModel):
    """데이터 수집 결과."""
    id: str            # 저장된 샘플의 고유 ID
    total: int         # 현재 수집된 전체 샘플 수


class SampleItem(BaseModel):
    """데이터 목록 조회 시 샘플 하나의 정보."""
    id: str
    question: str
    label: str
    audio_path: str


class DataListResponse(BaseModel):
    samples: List[SampleItem]
    total: int


class TrainStatusResponse(BaseModel):
    """학습 상태 응답."""
    status: str        # "idle" | "running" | "done" | "error"
    logs: List[str]    # 최근 로그 줄들
    sample_count: int  # 현재 수집된 샘플 수


# ─────────────────────────────────────────────────────────────────────────────
# STT 엔드포인트 (기존 기능)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """메인 UI 페이지를 반환합니다."""
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health():
    """서버 상태 확인용 엔드포인트."""
    return {"status": "ok"}


@app.post("/stt", response_model=STTResponse)
async def stt_baseline(audio: UploadFile = File(...)):
    """기본 STT — 음성만으로 변환."""
    _validate_audio(audio)
    audio_bytes = await audio.read()
    try:
        result = get_model().transcribe(audio_bytes)
    except Exception as e:
        logger.exception("Transcription error")
        raise HTTPException(status_code=500, detail=str(e))
    return STTResponse(**result)


@app.post("/stt/context", response_model=STTResponse)
async def stt_context(
    audio: UploadFile = File(...),
    question: str = Form(...),
):
    """문맥 인식 STT — 질문 + 음성으로 변환."""
    _validate_audio(audio)
    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    audio_bytes = await audio.read()
    try:
        result = get_model().transcribe(audio_bytes, question=question.strip())
    except Exception as e:
        logger.exception("Transcription error")
        raise HTTPException(status_code=500, detail=str(e))
    return STTResponse(**result)


@app.post("/stt/compare", response_model=CompareResponse)
async def stt_compare(
    audio: UploadFile = File(...),
    question: str = Form(...),
):
    """같은 음성에 대해 기본 모드와 문맥 모드를 동시에 실행합니다."""
    _validate_audio(audio)
    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    audio_bytes = await audio.read()
    model = get_model()
    try:
        baseline_result = model.transcribe(audio_bytes)
        context_result  = model.transcribe(audio_bytes, question=question.strip())
    except Exception as e:
        logger.exception("Transcription error")
        raise HTTPException(status_code=500, detail=str(e))
    return CompareResponse(
        baseline=STTResponse(**baseline_result),
        context=STTResponse(**context_result),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 수집 엔드포인트
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/data/collect", response_model=CollectResponse)
async def collect_sample(
    audio: UploadFile = File(...),
    question: str = Form(""),   # 콜봇 질문 (선택)
    label: str = Form(...),     # 올바른 정답 텍스트 (필수)
):
    """
    학습 데이터 1건을 저장합니다.

    저장 내용:
    - 오디오 파일: data/audio/{id}.wav
    - 메타데이터: data/dataset.jsonl 에 한 줄 추가

    [왜 WAV로 저장하나?]
    WAV는 무손실 포맷이라 학습 품질에 유리합니다.
    브라우저에서 이미 WAV로 변환해서 보내주기 때문에 그대로 저장합니다.
    """
    if not label.strip():
        raise HTTPException(status_code=422, detail="label(정답)을 입력하세요.")

    _validate_audio(audio)
    audio_bytes = await audio.read()

    # 고유 ID 생성 (UUID의 앞 8자리)
    sample_id = str(uuid.uuid4())[:8]

    # 오디오 파일 저장
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = AUDIO_DIR / f"{sample_id}.wav"
    audio_path.write_bytes(audio_bytes)

    # JSONL에 메타데이터 추가
    record = {
        "id":         sample_id,
        "question":   question.strip(),
        "audio_path": str(audio_path),
        "label":      label.strip(),
    }
    DATASET_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = _count_samples()
    logger.info(f"샘플 저장: id={sample_id}, label='{label.strip()}', 누적={total}")
    return CollectResponse(id=sample_id, total=total)


@app.get("/data/list", response_model=DataListResponse)
async def list_samples():
    """수집된 학습 데이터 전체 목록을 반환합니다."""
    samples = _load_samples()
    return DataListResponse(
        samples=[SampleItem(**s) for s in samples],
        total=len(samples),
    )


@app.delete("/data/{sample_id}")
async def delete_sample(sample_id: str):
    """
    특정 샘플을 삭제합니다.
    JSONL에서 해당 줄을 제거하고 오디오 파일도 삭제합니다.
    """
    samples = _load_samples()
    target = next((s for s in samples if s["id"] == sample_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"샘플을 찾을 수 없습니다: {sample_id}")

    # 오디오 파일 삭제
    audio_path = Path(target["audio_path"])
    if audio_path.exists():
        audio_path.unlink()

    # JSONL에서 해당 줄 제거 후 다시 씁니다
    remaining = [s for s in samples if s["id"] != sample_id]
    with open(DATASET_JSONL, "w", encoding="utf-8") as f:
        for s in remaining:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"샘플 삭제: id={sample_id}")
    return {"deleted": sample_id, "remaining": len(remaining)}


# ─────────────────────────────────────────────────────────────────────────────
# 모델 선택 엔드포인트
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/models/list")
async def list_models():
    """
    사용 가능한 모델 목록을 반환합니다.

    포함 항목:
    - 기본 모델 (Qwen3-ASR-0.6B, HuggingFace)
    - finetuned/ 아래에서 config.json이 있는 모든 run_* 폴더
    """
    models = [
        {
            "id":   BASE_MODEL_ID,
            "name": "Base Model (Qwen3-ASR-0.6B)",
            "type": "base",
            "path": BASE_MODEL_ID,
        }
    ]

    if FINETUNED_DIR.exists():
        # run_YYYYMMDD_HHMMSS 폴더를 최신순으로 정렬해서 추가합니다
        for d in sorted(FINETUNED_DIR.iterdir(), reverse=True):
            if d.is_dir() and (d / "config.json").exists():
                models.append({
                    "id":   str(d),
                    "name": f"Fine-tuned: {d.name}",
                    "type": "finetuned",
                    "path": str(d),
                })

    return {
        "models":  models,
        "current": get_current_model_id(),
    }


@app.post("/models/select")
async def select_model(model_id: str = Form(...)):
    """
    사용할 모델을 교체합니다.

    즉시 모델을 로드하지 않고, 다음 STT 요청 때 자동으로 로드합니다.
    (모델 로드에 수십 초가 걸리므로 백그라운드에서 처리)
    """
    # 유효성 검사: 기본 모델이거나 실제 존재하는 폴더여야 합니다
    if model_id != BASE_MODEL_ID and not (Path(model_id) / "config.json").exists():
        raise HTTPException(status_code=404, detail=f"모델을 찾을 수 없습니다: {model_id}")

    set_model(model_id)
    logger.info(f"모델 선택: {model_id}")
    return {"selected": model_id}


# ─────────────────────────────────────────────────────────────────────────────
# 학습 엔드포인트
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/train/start")
async def start_training(
    epochs: int   = Form(3),
    lr:     float = Form(1e-5),
):
    """
    파인튜닝을 시작합니다. train.py를 별도 프로세스로 실행합니다.

    [왜 별도 프로세스인가?]
    학습은 수 분~수십 분이 걸립니다. 같은 프로세스에서 실행하면
    그동안 서버가 다른 요청을 처리하지 못합니다.
    별도 프로세스로 실행하면 서버는 계속 동작하면서 학습 상태만 조회할 수 있습니다.
    """
    global _train_proc, _train_logs

    # 이미 학습 중이면 중복 실행을 막습니다
    if _train_proc is not None and _train_proc.poll() is None:
        raise HTTPException(status_code=400, detail="학습이 이미 실행 중입니다.")

    sample_count = _count_samples()
    if sample_count == 0:
        raise HTTPException(status_code=400, detail="학습 데이터가 없습니다. 먼저 데이터를 수집하세요.")

    _train_logs = []

    # train.py를 서브프로세스로 실행합니다
    # sys.executable: 현재 파이썬 인터프리터 경로 (가상환경의 python)
    cmd = [
        sys.executable, "train.py",
        "--data",   str(DATASET_JSONL),
        "--epochs", str(epochs),
        "--lr",     str(lr),
    ]
    _train_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # stderr를 stdout으로 합칩니다
        text=True,
        bufsize=1,                 # 줄 단위 버퍼링
    )

    # 백그라운드 스레드에서 로그를 읽습니다
    threading.Thread(
        target=_read_proc_output,
        args=(_train_proc,),
        daemon=True,  # 메인 프로세스 종료 시 같이 종료됩니다
    ).start()

    logger.info(f"학습 시작: {sample_count}개 샘플, {epochs}에폭, lr={lr}")
    return {"status": "started", "sample_count": sample_count}


@app.get("/train/status", response_model=TrainStatusResponse)
async def train_status():
    """
    현재 학습 상태와 로그를 반환합니다.
    UI에서 2초마다 폴링(polling)합니다.

    [폴링이란?]
    "지금 완료됐어?" 를 주기적으로 물어보는 방식입니다.
    """
    global _train_proc

    if _train_proc is None:
        status = "idle"
    elif _train_proc.poll() is None:
        # poll()이 None이면 프로세스가 아직 실행 중입니다
        status = "running"
    else:
        # poll()이 반환 코드를 돌려주면 프로세스가 종료된 것입니다
        status = "done" if _train_proc.returncode == 0 else "error"

    with _train_lock:
        recent_logs = list(_train_logs[-100:])  # 최근 100줄만 반환

    return TrainStatusResponse(
        status=status,
        logs=recent_logs,
        sample_count=_count_samples(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────

_ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3", "audio/ogg", "audio/flac",
    "application/octet-stream",
}


def _validate_audio(audio: UploadFile):
    """업로드된 파일이 허용된 오디오 형식인지 확인합니다."""
    if audio.content_type and audio.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"지원하지 않는 파일 형식: {audio.content_type}",
        )


def _load_samples() -> List[dict]:
    """JSONL 파일에서 모든 샘플을 읽어 리스트로 반환합니다."""
    if not DATASET_JSONL.exists():
        return []
    samples = []
    with open(DATASET_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def _count_samples() -> int:
    """수집된 샘플 수를 반환합니다."""
    return len(_load_samples())


# ─────────────────────────────────────────────────────────────────────────────
# 직접 실행할 때의 진입점
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
