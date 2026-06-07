import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from api_server.main import app
client = TestClient(app)
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
def test_root():
    response = client.get("/")
    assert response.status_code == 200
def test_predict():
    event = {
        "pipeline_id": "test-123",
        "build_number": 1,
        "status": "running",
        "stage": "build",
        "code_changes": {"files": ["a.py"], "additions": 100},
        "dependencies": {"updated": ["numpy"]}
    }
    response = client.post("/api/v1/predict", json=event)
    assert response.status_code == 200
    data = response.json()
    assert "risk_level" in data
def test_pipeline_failure():
    event = {
        "pipeline_id": "test-456",
        "build_number": 2,
        "status": "failed",
        "stage": "test",
        "error_message": "Connection timeout error"
    }
    response = client.post("/api/v1/pipeline/event", json=event)
    assert response.status_code == 200
    data = response.json()
    assert "analysis" in data
def test_heal():
    request = {
        "pipeline_id": "test-789",
        "error_log": "ERROR: Module not found: requests",
        "context": {"stage": "build"}
    }
    response = client.post("/api/v1/heal", json=request)
    assert response.status_code == 200
    assert response.json()["success"] in [True, False]
def test_stats():
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_runs" in data
    assert "success_rate" in data
print("All tests defined! Run with: pytest tests/test_api.py -v")
