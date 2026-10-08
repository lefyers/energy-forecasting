from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["features"] == 17


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "timestamp": "2018-08-03T01:00:00",
            "values": [15000.0] * 169,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["timestamp"] == "2018-08-03T01:00:00"
    assert isinstance(data["forecast"], float)


def test_prediction_requires_history():
    response = client.post(
        "/predict",
        json={
            "timestamp": "2018-08-03T01:00:00",
            "values": [15000.0] * 100,
        },
    )

    assert response.status_code == 422