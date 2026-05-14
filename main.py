"""
[이 파일의 역할]
FastAPI로 만든 웹 서버입니다.
브라우저나 앱이 음성 파일을 보내면, AI 모델로 텍스트로 변환(STT)한 결과를 돌려줍니다.

[STT란?]
STT(Speech-To-Text)는 사람이 말한 음성을 텍스트로 변환하는 기술입니다.
예: "안녕하세요" 라고 말한 녹음 파일 → "안녕하세요" 라는 문자열

[API란?]
API(Application Programming Interface)는 프로그램끼리 대화하는 방법입니다.
이 서버는 HTTP API를 제공합니다 — 브라우저가 요청을 보내면 서버가 결과를 JSON으로 돌려줍니다.

[제공하는 엔드포인트(URL 경로)]
POST /stt          — 음성 파일만 보내면 기본 STT 결과를 반환
POST /stt/context  — 질문 + 음성 파일을 보내면 문맥 인식 STT 결과를 반환
POST /stt/compare  — 위 두 가지를 동시에 실행하여 결과를 나란히 비교
GET  /health       — 서버가 살아있는지 확인하는 헬스체크
"""

# ─────────────────────────────────────────────────────────────────────────────
# 라이브러리(외부 도구) 불러오기
# ─────────────────────────────────────────────────────────────────────────────

import logging  # 서버에서 일어나는 일들을 콘솔에 기록(로그)하는 표준 도구

# FastAPI 관련 도구들
# - FastAPI  : 웹 서버 프레임워크. URL 경로를 정의하고 요청·응답을 처리합니다.
# - File     : 업로드된 파일을 받을 때 사용하는 선언자
# - Form     : HTML 폼(텍스트 입력)을 받을 때 사용하는 선언자
# - UploadFile: 파일 업로드 객체. 파일 이름, 타입, 내용이 들어 있습니다.
# - HTTPException: 오류가 생겼을 때 HTTP 에러 코드와 메시지를 돌려주는 도구
from fastapi import FastAPI, File, Form, UploadFile, HTTPException

# HTMLResponse: 일반 텍스트/JSON이 아닌 HTML 파일을 응답으로 돌려줄 때 사용
from fastapi.responses import HTMLResponse

# StaticFiles: CSS, JS, 이미지 등 정적 파일(바뀌지 않는 파일)을 서빙하는 도구
from fastapi.staticfiles import StaticFiles

# BaseModel: 응답 데이터의 형태(스키마)를 정의할 때 사용하는 Pydantic 클래스
# Pydantic은 데이터 유효성 검사를 자동으로 해줍니다.
# 예: text 필드에 숫자가 들어오면 자동으로 오류를 냅니다.
from pydantic import BaseModel

# 우리가 만든 model.py에서 AI 모델을 가져오는 함수
from model import get_model


# ─────────────────────────────────────────────────────────────────────────────
# 로깅(로그 기록) 설정
# ─────────────────────────────────────────────────────────────────────────────

# 서버가 실행되면 콘솔에 어떤 일이 일어나고 있는지 출력됩니다.
# level=logging.INFO : INFO 이상의 중요도인 메시지만 출력 (DEBUG는 너무 상세해서 제외)
# format="..." : "INFO main: 어쩌구" 형태로 출력
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# 이 파일 전용 로거. logger.info("...") 를 호출하면 콘솔에 출력됩니다.
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 앱 생성
# ─────────────────────────────────────────────────────────────────────────────

# app 이라는 이름으로 FastAPI 서버를 만듭니다.
# title/description/version은 자동 생성되는 API 문서(http://localhost:8001/docs)에 표시됩니다.
app = FastAPI(
    title="Qwen3-ASR Contextual STT",
    description="Callbot STT accuracy experiment: baseline vs. context-aware transcription",
    version="1.0.0",
)

# /static 경로로 들어오는 요청은 로컬의 "static" 폴더에서 파일을 찾아 돌려줍니다.
# 예: http://localhost:8001/static/index.html → 로컬 static/index.html 파일
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# 응답 스키마(Response Schema) 정의
# ─────────────────────────────────────────────────────────────────────────────
# 스키마란 데이터의 '모양'을 미리 정의하는 것입니다.
# 클라이언트(브라우저)가 받는 JSON의 구조를 명확히 하기 위해 사용합니다.

class STTResponse(BaseModel):
    """
    STT 결과 하나를 담는 데이터 구조입니다.

    예시 JSON:
    {
        "text": "안녕하세요 주문하시겠어요",
        "latency_sec": 1.234,
        "mode": "baseline"
    }
    """
    text: str          # 음성을 텍스트로 변환한 결과 문자열
    latency_sec: float # 처리에 걸린 시간(초). 성능을 측정하기 위해 기록합니다.
    mode: str          # "baseline"(기본) 또는 "context"(문맥 인식) 중 어느 모드인지


class CompareResponse(BaseModel):
    """
    /stt/compare 엔드포인트의 응답 구조입니다.
    baseline과 context 결과를 동시에 담아서 비교할 수 있게 합니다.

    예시 JSON:
    {
        "baseline": { "text": "이천원이요", "latency_sec": 1.1, "mode": "baseline" },
        "context":  { "text": "이천 원이요", "latency_sec": 1.3, "mode": "context" }
    }
    """
    baseline: STTResponse  # 기본 STT 결과
    context: STTResponse   # 문맥 인식 STT 결과


# ─────────────────────────────────────────────────────────────────────────────
# 라우트(Route) 정의
# ─────────────────────────────────────────────────────────────────────────────
# 라우트 = "이 URL로 요청이 오면 이 함수를 실행하라" 는 규칙
# @app.get(...)  → GET 방식 요청 (데이터를 조회할 때)
# @app.post(...) → POST 방식 요청 (데이터를 전송할 때. 파일 업로드는 항상 POST)

@app.get("/", response_class=HTMLResponse)
async def index():
    """
    http://localhost:8001/ 에 접속하면 static/index.html 파일을 돌려줍니다.
    즉, 브라우저에서 바로 UI를 볼 수 있게 해주는 진입점(entry point)입니다.
    """
    # open()으로 HTML 파일을 읽어서 그대로 브라우저에 전달합니다.
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health():
    """
    서버가 정상적으로 실행 중인지 확인하는 엔드포인트입니다.
    모니터링 시스템이나 배포 환경에서 "이 서버 살아있어?" 를 체크할 때 사용합니다.
    항상 {"status": "ok"} 를 반환합니다.
    """
    return {"status": "ok"}


@app.post("/stt", response_model=STTResponse)
async def stt_baseline(audio: UploadFile = File(...)):
    """
    [기본 STT 엔드포인트]
    음성 파일만 받아서 텍스트로 변환합니다. 문맥(질문) 없이 순수하게 음성만 인식합니다.

    요청 형식:
        Content-Type: multipart/form-data
        audio: (파일)

    응답 형식:
        { "text": "...", "latency_sec": 1.23, "mode": "baseline" }

    [async/await 이란?]
    파일 읽기, 네트워크 통신처럼 "기다려야 하는" 작업을 처리하는 방식입니다.
    await 키워드가 붙은 작업이 끝날 때까지 기다리되, 그동안 다른 요청도 처리할 수 있습니다.
    """
    # 업로드된 파일의 형식(타입)이 음성 파일인지 확인합니다.
    _validate_audio(audio)

    # 파일의 실제 내용을 바이트(bytes) 형태로 읽어옵니다.
    # await를 쓰는 이유: 파일 읽기는 I/O 작업이라 완료될 때까지 기다려야 하기 때문
    audio_bytes = await audio.read()

    try:
        # AI 모델을 불러와서 음성을 텍스트로 변환합니다.
        # get_model()은 최초 1회만 모델을 로드하고 이후엔 캐시된 것을 반환합니다.
        result = get_model().transcribe(audio_bytes)
    except Exception as e:
        # 변환 도중 오류가 생기면 로그에 기록하고 500 에러를 클라이언트에 전달합니다.
        # HTTP 500 = "서버 내부 오류"
        logger.exception("Transcription error")
        raise HTTPException(status_code=500, detail=str(e))

    # result는 {"text": ..., "latency_sec": ..., "mode": ...} 형태의 딕셔너리입니다.
    # **result 는 딕셔너리를 풀어서 키=값 형태로 전달하는 파이썬 문법입니다.
    # 즉, STTResponse(text=..., latency_sec=..., mode=...) 와 동일합니다.
    return STTResponse(**result)


@app.post("/stt/context", response_model=STTResponse)
async def stt_context(
    audio: UploadFile = File(...),   # 음성 파일 (필수)
    question: str = Form(...),       # 콜봇이 사용자에게 물어본 질문 텍스트 (필수)
):
    """
    [문맥 인식 STT 엔드포인트]
    질문 텍스트 + 음성 파일을 함께 받아서 텍스트로 변환합니다.

    [문맥(Context)이 왜 필요한가?]
    콜봇이 "몇 개 주문하시겠어요?" 라고 물었고, 사용자가 "두 개요" 라고 답했다면,
    모델이 질문을 알고 있으면 "두 개요" 를 더 정확하게 인식할 수 있습니다.
    특히 숫자, 고유명사, 전문용어 등에서 효과가 큽니다.

    요청 형식:
        Content-Type: multipart/form-data
        audio: (파일)
        question: "고객님 성함이 어떻게 되세요?"
    """
    _validate_audio(audio)

    # 질문이 빈 문자열이면 400 에러를 반환합니다.
    # strip()은 앞뒤 공백을 제거합니다. "  " 같은 공백만 있는 입력도 걸러냅니다.
    # HTTP 422 = "요청 데이터가 올바르지 않음(Unprocessable Entity)"
    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    audio_bytes = await audio.read()

    try:
        # question을 함께 전달해서 문맥을 알려줍니다.
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
    """
    [비교 STT 엔드포인트]
    같은 음성 파일에 대해 기본 모드와 문맥 인식 모드를 동시에 실행합니다.
    결과를 나란히 비교해서 "문맥을 주는 게 실제로 얼마나 도움이 되는지" 를 실험합니다.

    이 엔드포인트가 이 프로젝트의 핵심입니다.
    같은 음성을 두 가지 방식으로 처리하고 결과를 비교하는 게 목표이기 때문입니다.
    """
    _validate_audio(audio)
    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    # 파일은 한 번만 읽고, 읽은 바이트를 두 번 재사용합니다.
    # (파일 포인터는 한번 읽으면 끝으로 가기 때문에 bytes로 미리 저장합니다)
    audio_bytes = await audio.read()

    # 모델을 한 번만 가져옵니다. (두 번 호출해도 같은 객체가 반환됩니다)
    model = get_model()

    try:
        # 1. 기본 모드: 문맥 없이 음성만으로 변환
        baseline_result = model.transcribe(audio_bytes)
        # 2. 문맥 모드: 질문을 힌트로 주고 변환
        context_result = model.transcribe(audio_bytes, question=question.strip())
    except Exception as e:
        logger.exception("Transcription error")
        raise HTTPException(status_code=500, detail=str(e))

    # 두 결과를 CompareResponse에 담아서 반환합니다.
    return CompareResponse(
        baseline=STTResponse(**baseline_result),
        context=STTResponse(**context_result),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼(보조) 함수
# ─────────────────────────────────────────────────────────────────────────────

# 허용하는 오디오 파일 형식 목록 (MIME 타입)
# MIME 타입이란 파일의 종류를 나타내는 표준 문자열입니다.
# 브라우저는 파일을 보낼 때 "이 파일은 audio/wav 입니다" 라고 알려줍니다.
_ALLOWED_CONTENT_TYPES = {
    "audio/wav",               # WAV 파일
    "audio/x-wav",             # WAV의 다른 표기
    "audio/wave",              # WAV의 또 다른 표기
    "audio/mpeg",              # MP3 파일
    "audio/mp3",               # MP3의 다른 표기
    "audio/ogg",               # OGG 파일 (오픈소스 음성 포맷)
    "audio/flac",              # FLAC 파일 (무손실 압축 포맷)
    "application/octet-stream",  # 타입을 모를 때 브라우저가 쓰는 기본값. 허용해둡니다.
}


def _validate_audio(audio: UploadFile):
    """
    업로드된 파일이 음성 파일인지 검사합니다.
    함수 이름 앞의 _ (언더스코어)는 "이 함수는 이 파일 내부에서만 쓴다" 는 관례입니다.

    [content_type이란?]
    브라우저가 파일을 보낼 때 함께 전달하는 파일 형식 정보입니다.
    content_type이 None인 경우(알 수 없는 경우)는 통과시킵니다.

    [HTTP 415란?]
    "Unsupported Media Type" — 서버가 처리할 수 없는 파일 형식이라는 의미입니다.
    """
    if audio.content_type and audio.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {audio.content_type}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 직접 실행할 때의 진입점
# ─────────────────────────────────────────────────────────────────────────────

# `python main.py` 로 직접 실행했을 때만 아래 코드가 실행됩니다.
# `uvicorn main:app` 으로 실행하면 이 블록은 무시됩니다.
# run.sh에서는 uvicorn을 직접 실행하므로 이 블록은 사용되지 않습니다.
if __name__ == "__main__":
    import uvicorn
    # uvicorn은 FastAPI 앱을 실행하는 고성능 웹 서버입니다.
    # host="0.0.0.0" : 이 컴퓨터의 모든 네트워크 인터페이스에서 접속 허용
    # port=8000      : 8000번 포트에서 대기
    # reload=False   : 코드 변경 시 자동 재시작 비활성화 (운영 환경)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
