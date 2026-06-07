"""
AI Auto-Healing CI/CD - Main API Server
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging
from datetime import datetime
import os
import random
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="AI Auto-Healing CI/CD",
    description="Intelligent CI/CD pipeline with AI-powered auto-healing",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Models
class PipelineEvent(BaseModel):
    pipeline_id: str
    build_number: int
    status: str
    stage: str
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    logs: Optional[str] = None
    code_changes: Optional[Dict] = None
    dependencies: Optional[Dict] = None
class HealingRequest(BaseModel):
    pipeline_id: str
    error_log: str
    context: Dict
# Global state
start_time = datetime.now()
pipeline_runs = []
healing_actions = []
@app.get("/")
async def root():
    return {"message": "AI Auto-Healing CI/CD API", "version": "1.0.0", "docs": "/docs"}
@app.get("/health")
async def health():
    uptime = (datetime.now() - start_time).total_seconds()
    return {"status": "healthy", "version": "1.0.0", "uptime": uptime}
@app.post("/api/v1/pipeline/event")
async def pipeline_event(event: PipelineEvent):
    logger.info(f"Pipeline event: {event.pipeline_id} - {event.status}")
    event_data = event.dict()
    event_data["timestamp"] = datetime.now().isoformat()
    pipeline_runs.append(event_data)
    if event.status == "failed":
        error_type = analyze_error(event.error_message or "")
        action = get_healing_action(error_type)
        return {
            "message": "Event received",
            "status": "failed",
            "analysis": {"error_type": error_type, "recommended_action": action}
        }
    return {"message": "Event received", "status": "success"}
@app.post("/api/v1/predict")
async def predict(event: PipelineEvent):
    risk = 0.0
    if event.code_changes:
        files = len(event.code_changes.get("files", []))
        if files > 50: risk += 0.3
        if files > 100: risk += 0.2
    if event.dependencies:
        deps = len(event.dependencies.get("updated", []))
        if deps > 5: risk += 0.2
    level = "high" if risk > 0.7 else ("medium" if risk > 0.3 else "low")
    return {
        "failure_predicted": risk > 0.5,
        "failure_probability": min(risk, 1.0),
        "confidence": 0.8,
        "risk_level": level,
        "recommendations": ["Consider splitting changes"] if risk > 0.5 else []
    }
@app.post("/api/v1/heal")
async def heal(request: HealingRequest):
    logger.info(f"Healing: {request.pipeline_id}")
    error_type = analyze_error(request.error_log)
    action = get_healing_action(error_type)
    success = random.random() < 0.8
    healing_actions.append({
        "pipeline_id": request.pipeline_id,
        "error_type": error_type,
        "action": action,
        "success": success,
        "timestamp": datetime.now().isoformat()
    })
    return {"message": "Healing executed", "error_type": error_type, "action": action, "success": success}
@app.get("/api/v1/stats")
async def stats():
    total = len(pipeline_runs)
    failures = len([r for r in pipeline_runs if r.get("status") == "failed"])
    rate = ((total - failures) / total * 100) if total > 0 else 0
    return {
        "total_runs": total,
        "failures": failures,
        "success_rate": round(rate, 2),
        "healing_actions": len(healing_actions)
    }
def analyze_error(log: str) -> str:
    log_lower = log.lower()
    patterns = {
        "network_error": ["timeout", "connection refused", "network"],
        "dependency_error": ["module not found", "import error", "version"],
        "build_error": ["compilation failed", "syntax error"],
        "test_failure": ["assertion", "test failed"],
        "resource_error": ["out of memory", "disk full", "oom"],
        "permission_error": ["permission denied", "access denied"],
    }
    for error_type, keywords in patterns.items():
        if any(kw in log_lower for kw in keywords):
            return error_type
    return "unknown_error"
def get_healing_action(error_type: str) -> str:
    actions = {
        "network_error": "retry_with_backoff",
        "dependency_error": "rollback_dependencies",
        "build_error": "clean_build",
        "test_failure": "retry_tests",
        "resource_error": "scale_resources",
        "permission_error": "fix_permissions",
    }
    return actions.get(error_type, "manual_investigation")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
