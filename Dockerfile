# ── 베이스 이미지 ────────────────────────────────────────────────────────────
# python:3.11-slim: 파이썬만 있는 가벼운 Debian 이미지
# 3.13 대신 3.11을 쓰는 이유: PyTorch/librosa 등 ML 패키지가 3.11에서 가장 안정적
FROM python:3.11-slim

# ── 작업 디렉토리 설정 ────────────────────────────────────────────────────────
WORKDIR /app

# ── 시스템 패키지 설치 ────────────────────────────────────────────────────────
# libsndfile1 : soundfile/librosa가 오디오 파일을 읽을 때 필요한 C 라이브러리
# libgomp1    : PyTorch가 CPU 병렬 연산에 사용하는 OpenMP 라이브러리
# git         : pip가 git+https:// 형식 패키지를 설치할 때 필요
# ffmpeg      : 다양한 오디오 포맷 변환 지원 (선택적이지만 있으면 안전)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libgomp1 \
        git \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# rm -rf /var/lib/apt/lists/* : apt 캐시를 지워서 이미지 크기를 줄입니다

# ── Python 의존성 설치 ────────────────────────────────────────────────────────
# requirements.txt 먼저 복사하는 이유:
#   소스 코드가 바뀌어도 requirements.txt가 그대로면 이 레이어는 캐시를 재사용합니다.
#   → docker build가 훨씬 빨라집니다.
COPY requirements.txt .

# CPU 전용 PyTorch 설치 (GPU 버전 대비 ~1.5GB 절약)
# --index-url: CUDA 버전 대신 CPU-only 휠을 내려받을 주소를 지정합니다
RUN pip install --no-cache-dir \
        torch torchaudio \
        --index-url https://download.pytorch.org/whl/cpu

# 나머지 패키지 설치 (torch는 이미 설치됐으므로 중복 방지용 --no-deps 불필요)
RUN pip install --no-cache-dir -r requirements.txt

# ── 모델 사전 다운로드 ────────────────────────────────────────────────────────
# 빌드 시점에 HuggingFace에서 모델을 내려받아 이미지에 포함시킵니다.
# 장점: 컨테이너 실행 후 첫 요청이 바로 빠릅니다.
# 단점: 이미지 크기가 ~1.4GB 증가합니다.
#
# 환경변수 설명:
#   HF_HOME      : HuggingFace 캐시 저장 위치 (기본값과 동일하지만 명시)
#   TRANSFORMERS_OFFLINE=0 : 온라인 다운로드 허용
ENV HF_HOME=/root/.cache/huggingface

RUN python - <<'EOF'
import torch, qwen_asr                       # qwen_asr 임포트가 AutoModel에 모델 등록
from qwen_asr import Qwen3ASRModel
print("모델 다운로드 시작...")
Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    torch_dtype=torch.float32,
    device_map="cpu",
)
print("모델 다운로드 완료")
EOF

# ── 소스 코드 복사 ────────────────────────────────────────────────────────────
# 소스 코드는 마지막에 복사합니다.
# 코드 수정 시 위의 패키지/모델 레이어를 재사용할 수 있습니다.
COPY . .

# ── 포트 및 볼륨 선언 ─────────────────────────────────────────────────────────
EXPOSE 8001

# 외부에서 마운트할 디렉토리를 선언합니다.
# VOLUME 선언은 문서 역할이며, docker-compose.yml에서 실제 마운트를 정의합니다.
VOLUME ["/app/data", "/app/finetuned"]

# ── 서버 실행 ─────────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
