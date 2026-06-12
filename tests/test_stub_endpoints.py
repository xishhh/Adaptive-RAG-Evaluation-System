from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_query_not_implemented():
    response = client.post("/query", json={"question": "What is RAG?"})
    assert response.status_code == 501


def test_evaluate_not_implemented():
    response = client.post("/evaluate", json={"dataset_path": "qa_pairs.json"})
    assert response.status_code == 501


def test_metrics_not_implemented():
    response = client.get("/metrics")
    assert response.status_code == 501