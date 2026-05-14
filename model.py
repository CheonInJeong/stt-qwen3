"""
[이 파일의 역할]
AI 음성 인식 모델(Qwen3-ASR)을 불러오고, 음성을 텍스트로 변환하는 기능을 담당합니다.
main.py(서버)와 AI 모델 사이의 다리 역할을 합니다.

[Qwen3-ASR이란?]
중국 알리바바가 만든 음성 인식(STT) AI 모델입니다.
0.6B는 매개변수(parameter)가 6억 개라는 뜻입니다.
매개변수가 많을수록 더 정확하지만 메모리와 연산이 더 많이 필요합니다.

[Baseline vs Contextual 모드]
- Baseline (기본 모드)  : 음성만 보고 텍스트로 변환
- Contextual (문맥 모드): 콜봇이 어떤 질문을 했는지 알려주고 변환
  → 같은 발음이라도 문맥에 따라 더 정확하게 인식합니다.
  예: "이천" → 문맥 없으면 지명일 수도, 숫자 2000일 수도 있음
"""

# ─────────────────────────────────────────────────────────────────────────────
# 라이브러리(외부 도구) 불러오기
# ─────────────────────────────────────────────────────────────────────────────

import io       # 파일을 디스크에 저장하지 않고 메모리에서 처리할 때 사용
import time     # 처리 시간을 측정할 때 사용
import logging  # 콘솔에 진행 상황을 출력하는 표준 도구

import numpy as np  # 수치 계산 라이브러리. 음성 데이터는 숫자 배열로 표현됩니다.
import torch        # PyTorch: AI 모델을 실행하는 딥러닝 프레임워크
import librosa      # 음성 파일을 읽고 처리하는 전문 라이브러리

# Optional: 값이 있을 수도 있고 None(없음)일 수도 있는 타입을 표현합니다.
# 예: Optional[str] = str 또는 None
from typing import Optional

# Qwen3-ASR 공식 패키지에서 모델 클래스를 가져옵니다.
# 앞의 _ (언더스코어)는 "내부용으로 이름을 바꿔서 쓴다" 는 뜻입니다.
# 우리가 아래에 QwenASRModel이라는 래퍼(wrapper) 클래스를 따로 만들기 때문입니다.
from qwen_asr import Qwen3ASRModel as _Qwen3ASRModel

# 이 파일 전용 로거
logger = logging.getLogger(__name__)

# 사용할 모델의 HuggingFace 저장소 주소
# HuggingFace는 AI 모델을 공개적으로 공유하는 플랫폼입니다. (github와 비슷)
MODEL_ID = "Qwen/Qwen3-ASR-0.6B"


# ─────────────────────────────────────────────────────────────────────────────
# 모델 래퍼(Wrapper) 클래스
# ─────────────────────────────────────────────────────────────────────────────
# 래퍼(Wrapper)란? 복잡한 라이브러리를 우리 프로젝트에 맞게 감싸서
# 사용하기 쉽게 만든 클래스입니다.
# main.py가 AI 라이브러리의 복잡한 내부를 몰라도 되게 해줍니다.

class QwenASRModel:
    """
    Qwen3-ASR 모델을 감싸는 래퍼 클래스.
    main.py에서는 이 클래스의 transcribe() 메서드만 호출하면 됩니다.
    """

    def __init__(self, model_id: str = MODEL_ID):
        """
        클래스 생성자(Constructor). 객체를 만들 때 자동으로 실행됩니다.

        [파라미터]
        model_id: 사용할 모델의 식별자. 기본값은 위에서 정의한 MODEL_ID.
        """
        self.model_id = model_id  # 나중에 다른 메서드에서도 쓸 수 있게 저장
        self._load_model()        # 객체 생성과 동시에 모델을 로드합니다

    def _load_model(self):
        """
        AI 모델을 HuggingFace에서 다운로드하거나 캐시에서 불러옵니다.
        최초 실행 시 수백 MB의 파일을 다운로드하므로 시간이 걸릴 수 있습니다.
        이후엔 로컬 캐시(~/.cache/huggingface/)에서 빠르게 불러옵니다.

        [torch_dtype=torch.float32 이란?]
        모델의 숫자 정밀도를 설정합니다.
        - float32: 32비트 부동소수점. CPU에서 안정적으로 동작합니다.
        - bfloat16: 16비트. GPU에서 빠르지만 CPU에서 지원이 제한적입니다.

        [device_map="cpu" 이란?]
        모델을 CPU에서 실행하라는 의미입니다.
        GPU(그래픽카드)가 있으면 "cuda:0" 등으로 바꾸면 훨씬 빠릅니다.
        CPU는 느리지만 GPU 없이도 실행 가능합니다.
        """
        logger.info(f"Loading model: {self.model_id} (CPU)")

        self.model = _Qwen3ASRModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,  # CPU용 숫자 타입
            device_map="cpu",           # CPU에서 실행
        )

        logger.info("Model loaded successfully")

    def _load_audio(self, audio_bytes: bytes) -> np.ndarray:
        """
        음성 파일(바이트 데이터)을 AI 모델이 처리할 수 있는 숫자 배열로 변환합니다.

        [음성 파일이 숫자 배열이 되는 원리]
        소리는 공기의 진동입니다. 마이크는 이 진동을 시간에 따른 숫자(진폭)로 기록합니다.
        예: [0.01, 0.05, 0.12, 0.08, ...] 처럼 매우 많은 숫자의 연속입니다.

        [sr=16000 (샘플링 레이트)이란?]
        1초에 몇 개의 숫자로 소리를 표현하는가를 나타냅니다.
        16000Hz = 1초에 16,000개의 숫자.
        전화 품질의 음성에 적합한 표준값입니다.
        AI 음성 모델 대부분이 16000Hz를 기준으로 학습되어 있습니다.

        [mono=True 이란?]
        스테레오(좌우 2채널)를 모노(1채널)로 합칩니다.
        음성 인식에는 채널 구분이 필요 없기 때문입니다.

        [io.BytesIO 이란?]
        audio_bytes(메모리에 있는 데이터)를 파일처럼 읽을 수 있게 해주는 도구입니다.
        디스크에 파일을 저장하지 않고도 librosa가 처리할 수 있게 합니다.

        [반환값]
        np.ndarray: numpy 숫자 배열. AI 모델이 소비하는 형태입니다.
        float32   : 각 숫자가 32비트 실수임을 보장합니다.
        """
        audio, _ = librosa.load(
            io.BytesIO(audio_bytes),  # 바이트를 파일처럼 포장
            sr=16000,                 # 16kHz로 리샘플링(표준화)
            mono=True,                # 모노로 변환
        )
        # astype(np.float32): 숫자 타입을 float32로 명시적으로 변환
        return audio.astype(np.float32)

    def transcribe(
        self,
        audio_bytes: bytes,
        question: Optional[str] = None,
    ) -> dict:
        """
        음성을 텍스트로 변환(전사, Transcription)합니다.
        이 메서드가 이 클래스의 핵심 기능입니다.

        [파라미터]
        audio_bytes: 음성 파일의 원본 바이트 데이터
        question   : 콜봇이 사용자에게 물어본 질문(선택 사항)
                     None이면 기본 모드, 값이 있으면 문맥 인식 모드

        [반환값]
        딕셔너리(dict): { "text": ..., "latency_sec": ..., "mode": ... }

        [전사(Transcription)란?]
        음성을 텍스트로 적는 것을 전사라고 합니다.
        AI가 자동으로 하는 것을 자동 음성 인식(ASR)이라고 합니다.
        """
        # 시간 측정 시작. perf_counter()는 매우 정밀한 타이머입니다.
        t0 = time.perf_counter()

        # question이 있으면 "context" 모드, 없으면 "baseline" 모드로 기록
        mode = "context" if question else "baseline"

        # 바이트 데이터를 AI가 처리할 수 있는 숫자 배열로 변환
        audio = self._load_audio(audio_bytes)

        # Qwen3-ASR 모델로 음성을 전사합니다.
        # audio=(audio, 16000) : (숫자배열, 샘플링레이트) 튜플로 전달
        # context=question or "": question이 None이면 빈 문자열("")을 전달
        #   → or 연산자: 왼쪽이 None/False이면 오른쪽 값을 사용
        # language="Korean"을 반드시 지정해야 합니다. 두 가지 이유:
        #   1) 지정 안 하면 자동 감지 → 짧은 한국어 발음(예: "예")을 중국어로 오인식
        #   2) 지정 안 하면 출력 파싱 실패 시 context(질문 텍스트)가 그대로 결과로 반환됨
        #   language 지정 시 모델은 파싱 없이 한국어 텍스트만 직접 출력합니다.
        results = self.model.transcribe(
            audio=(audio, 16000),
            context=question or "",
            language="Korean",
        )

        # results는 리스트입니다. 우리는 파일 1개를 보냈으니 [0]번째 결과를 씁니다.
        # .text 속성에 인식된 텍스트가 있습니다.
        # .strip()은 앞뒤 공백/줄바꿈을 제거합니다.
        transcription = results[0].text.strip()

        # 시간 측정 종료. t0(시작)과의 차이가 처리 시간입니다.
        # round(..., 3): 소수점 3자리까지 반올림 (밀리초 단위 정밀도)
        latency = round(time.perf_counter() - t0, 3)

        # 결과를 콘솔에 출력합니다.
        logger.info(f"[{mode}] '{transcription}' ({latency}s)")

        # 딕셔너리로 결과를 반환합니다. main.py의 STTResponse가 이 구조를 받습니다.
        return {"text": transcription, "latency_sec": latency, "mode": mode}


# ─────────────────────────────────────────────────────────────────────────────
# 싱글톤(Singleton) 패턴: 모델을 딱 한 번만 로드하기
# ─────────────────────────────────────────────────────────────────────────────
# AI 모델은 로드하는 데 시간과 메모리가 많이 필요합니다.
# 요청이 들어올 때마다 매번 로드하면 너무 느립니다.
# 싱글톤 패턴은 "객체를 딱 하나만 만들고, 계속 재사용"하는 설계 방식입니다.

# ─────────────────────────────────────────────────────────────────────────────
# 싱글톤 + 동적 모델 교체
# ─────────────────────────────────────────────────────────────────────────────
# _current_model_id: 현재 사용할 모델 경로. 처음엔 기본 모델(Qwen3-ASR-0.6B).
# _model_instance  : 로드된 모델 객체. None이면 아직 로드 안 됨.

_model_instance: Optional[QwenASRModel] = None
_current_model_id: str = MODEL_ID


def get_model() -> QwenASRModel:
    """
    현재 선택된 모델 인스턴스를 반환합니다.
    처음 호출하거나 set_model() 이후 첫 호출이면 모델을 로드합니다.
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = QwenASRModel(model_id=_current_model_id)
    return _model_instance


def set_model(new_model_id: str) -> None:
    """
    사용할 모델을 교체합니다.

    기존 모델 인스턴스를 버리고 None으로 초기화합니다.
    다음 get_model() 호출 시 새 모델이 로드됩니다.

    [왜 즉시 로드하지 않나?]
    모델 로드는 수십 초가 걸립니다.
    API 요청이 왔을 때 로드하면 첫 번째 요청만 느리고 이후엔 빠릅니다.
    (lazy loading 패턴)
    """
    global _model_instance, _current_model_id
    _current_model_id = new_model_id
    _model_instance = None  # 다음 get_model() 호출 시 새로 로드됩니다
    logger.info(f"모델 교체 예정: {new_model_id}")


def get_current_model_id() -> str:
    """현재 선택된 모델 ID(경로)를 반환합니다."""
    return _current_model_id
