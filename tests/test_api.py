from fastapi.testclient import TestClient


def _client(trained_pipeline):
    # Import after the workspace fixture has re-pointed all paths.
    import importlib
    import jasmin.api.app as app_module
    importlib.reload(app_module)
    app_module.config = trained_pipeline["config"]
    return TestClient(app_module.app)


def test_health_reports_ready_model(trained_pipeline):
    client = _client(trained_pipeline)
    body = client.get("/health").json()
    assert body["model"]["status"] == "ready"


def test_predict_endpoint(trained_pipeline):
    client = _client(trained_pipeline)
    resp = client.get("/predict/reliance")  # case-insensitive
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "RELIANCE"
    assert body["direction"] in ("UP", "DOWN")
    assert "explanation" in body and "confidence" in body


def test_predict_unknown_symbol_404(trained_pipeline):
    client = _client(trained_pipeline)
    assert client.get("/predict/NOTASTOCK").status_code == 404


def test_models_and_predictions_endpoints(trained_pipeline):
    client = _client(trained_pipeline)
    assert isinstance(client.get("/models").json(), list)
    preds = client.get("/predictions").json()
    assert isinstance(preds, list)
