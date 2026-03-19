from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid
import time

# Import the modular orchestrator we built
from core_engine import KYCOrchestrator

app = FastAPI(title="Defensive KYC Security Gateway", version="1.0")

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[System] Initializing Modular KYC Engine...")
engine = KYCOrchestrator()
print("[System] All security layers operational.")

os.makedirs("temp_uploads", exist_ok=True)


# Define the exact JSON structure the frontend will receive
class StageResult(BaseModel):
    score: float
    passed: bool
    details: str


class KYCResponse(BaseModel):
    processing_time_seconds: float
    stage_1_sensor: StageResult
    stage_2_biological: StageResult
    stage_3_temporal_ai: StageResult
    final_decision: str


@app.post("/api/v1/audit_stream", response_model=KYCResponse)
async def audit_video_stream(video: UploadFile = File(...)):
    """
    Ingests a video payload and runs it through the 3-stage security gateway.
    """
    if not video.filename.endswith(('.mp4', '.avi', '.mov', '.webm')):
        raise HTTPException(status_code=400, detail="Unsupported media format.")

    temp_path = f"temp_uploads/{uuid.uuid4()}_{video.filename}"

    try:
        # 1. Save Payload
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        start_time = time.time()

        # 2. Execute Modular Engine
        results = engine.analyze_video(temp_path)

        proc_time = round(time.time() - start_time, 2)

        # 3. Format Response for the Audit Dashboard
        response = KYCResponse(
            processing_time_seconds=proc_time,
            stage_1_sensor=StageResult(
                score=results["replay_attack_score"],
                passed=not results["is_replay_attack"],
                details="Moiré frequency threshold check"
            ),
            stage_2_biological=StageResult(
                score=results["biological_bpm"],
                passed=results["is_lively"],
                details="rPPG heartbeat extraction (45-150 BPM target)"
            ),
            stage_3_temporal_ai=StageResult(
                score=results["ai_manipulation_score"],
                passed=not results["is_deepfake"],
                details="3D CNN temporal inconsistency check"
            ),
            final_decision="APPROVED"  # Will be overwritten if any stage fails
        )

        # 4. The Waterfall Logic (Determine Final Decision)
        if not response.stage_1_sensor.passed:
            response.final_decision = "DENIED: REPLAY_ATTACK_DETECTED"
        elif not response.stage_2_biological.passed:
            response.final_decision = "DENIED: NO_BIOLOGICAL_LIVENESS"
        elif not response.stage_3_temporal_ai.passed:
            response.final_decision = "DENIED: SYNTHETIC_MEDIA_DETECTED"

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine Failure: {str(e)}")

    finally:
        # Clean up the artifact to preserve storage
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/v1/system_status")
def health_check():
    """Endpoint for frontend to verify engine is online before sending video"""
    return {
        "status": "online",
        "deepfake_module_device": str(engine.deepfake_module.device)
    }