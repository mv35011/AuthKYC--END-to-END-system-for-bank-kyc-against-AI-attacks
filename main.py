from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
import uuid
import time
import cv2

from core_engine import KYCOrchestrator

app = FastAPI(title="Defensive KYC PAD Gateway", version="2.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

engine = KYCOrchestrator()
os.makedirs("temp_uploads", exist_ok=True)


class StageResult(BaseModel):
    score: float
    passed: bool
    details: str


class KYCResponse(BaseModel):
    processing_time_seconds: float
    telemetry_status: str
    stage_1_sensor_prnu: StageResult
    stage_2_presentation_replay: StageResult
    stage_3_biological_rppg: StageResult
    stage_4_synthetic_ftca: StageResult
    final_decision: str


def analyze_stream_telemetry(video_path):
    """Checks for OBS/Virtual Camera frame dropping and unnatural metadata."""
    cap = cv2.VideoCapture(video_path)
    expected_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / expected_fps if expected_fps > 0 else 0
    cap.release()

    # Virtual cameras often write files with slightly corrupted headers
    # or perfectly rigid, unnatural integer frame rates.
    if expected_fps == 0 or duration == 0:
        return "WARNING: Corrupted Video Metadata (Possible Injection)"
    if expected_fps.is_integer():
        return "NOTICE: Rigid Frame Timing (Monitor for OBS)"
    return "OK: Natural Stream Variance"


# ── Frontend ──
@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


# ── Rewrite static asset paths (styles.css, app.js) ──
@app.get("/styles.css")
async def serve_css():
    return FileResponse("frontend/styles.css")

@app.get("/app.js")
async def serve_js():
    return FileResponse("frontend/app.js")


@app.post("/api/v1/audit_stream", response_model=KYCResponse)
async def audit_video_stream(video: UploadFile = File(...)):
    if not video.filename.endswith(('.mp4', '.avi', '.mov', '.webm')):
        raise HTTPException(status_code=400, detail="Unsupported media format.")

    temp_path = f"temp_uploads/{uuid.uuid4()}_{video.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        start_time = time.time()

        # Fast Pre-check: Telemetry
        telemetry = analyze_stream_telemetry(temp_path)

        # Heavy Compute: 4-Layer PAD Engine
        results = engine.analyze_video(temp_path)

        proc_time = round(time.time() - start_time, 2)

        response = KYCResponse(
            processing_time_seconds=proc_time,
            telemetry_status=telemetry,
            stage_1_sensor_prnu=StageResult(
                score=results["prnu_energy"],
                passed=not results["is_virtual_camera"],
                details="PRNU variance check for physical CMOS sensor"
            ),
            stage_2_presentation_replay=StageResult(
                score=results["moire_score"],
                passed=not results["is_replay_attack"],
                details="Moiré high-frequency screen grid check"
            ),
            stage_3_biological_rppg=StageResult(
                score=results["rppg_snr"],
                passed=results["is_lively"],
                details=f"CHROM rPPG SNR threshold (Pulse: {results['biological_bpm']:.1f} BPM)"
            ),
            stage_4_synthetic_ftca=StageResult(
                score=results["ai_manipulation_score"],
                passed=not results["is_deepfake"],
                details="FTCA Frequency-Temporal Cross-Attention check"
            ),
            final_decision="APPROVED"
        )

        # The Strict Waterfall Liveness Gate
        if not response.stage_1_sensor_prnu.passed:
            response.final_decision = "DENIED: VIRTUAL_CAMERA_INJECTION"
        elif not response.stage_2_presentation_replay.passed:
            response.final_decision = "DENIED: SCREEN_REPLAY_ATTACK"
        elif not response.stage_3_biological_rppg.passed:
            response.final_decision = "DENIED: BIOLOGICAL_LIVENESS_FAILED"
        elif not response.stage_4_synthetic_ftca.passed:
            response.final_decision = "DENIED: SYNTHETIC_AI_GENERATION"

        return response

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)