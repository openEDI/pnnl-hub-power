import json
import logging
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# server imports helics (via hub_federate) and uvicorn — both mocked in conftest.py
import server
from server import app, build_url


# ---------------------------------------------------------------------------
# build_url
# ---------------------------------------------------------------------------

def test_build_url_no_kubernetes_plain_url(caplog):
    """T21 — no Kubernetes service → plain URL, logs docker-compose."""
    with patch("server.kubernetes_service", return_value=None):
        with caplog.at_level(logging.INFO):
            url = build_url("myhost", 8080, ["sensor"])
    assert url == "http://myhost:8080/sensor"
    assert any("docker-compose environment" in r.message for r in caplog.records)


def test_build_url_kubernetes_namespaced_url(caplog):
    """T22 — Kubernetes service present → namespaced URL, logs kubernetes."""
    with patch("server.kubernetes_service", return_value="mynamespace"):
        with caplog.at_level(logging.INFO):
            url = build_url("myhost", 8080, ["sensor"])
    assert url == "http://myhost.mynamespace:8080/sensor"
    assert any("kubernetes environment" in r.message for r in caplog.records)


def test_build_url_multi_segment_path():
    """T23 — multi-segment endpoint path."""
    with patch("server.kubernetes_service", return_value=None):
        url = build_url("myhost", 9000, ["api", "v1", "data"])
    assert url == "http://myhost:9000/api/v1/data"


def test_build_url_empty_endpoint():
    """T24 — empty endpoint list → trailing slash."""
    with patch("server.kubernetes_service", return_value=None):
        url = build_url("myhost", 80, [])
    assert url == "http://myhost:80/"


# ---------------------------------------------------------------------------
# GET / — health check
# ---------------------------------------------------------------------------

def test_read_root_returns_200_with_hostname_and_ip():
    """T25 — returns 200 with hostname and host_ip keys."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "hostname" in data
    assert "host_ip" in data


# ---------------------------------------------------------------------------
# POST /run — run_model
# ---------------------------------------------------------------------------

def test_run_model_happy_path(tmp_path, monkeypatch):
    """T26 — happy path: sensor data fetched, sensors.json written, background task queued."""
    from unittest.mock import MagicMock
    sensors_file = tmp_path / "sensors.json"

    import builtins
    real_open = builtins.open

    def patched_open(path, mode="r", *args, **kwargs):
        if path == "sensors.json" and "w" in mode:
            return real_open(str(sensors_file), mode, *args, **kwargs)
        return real_open(path, mode, *args, **kwargs)

    mock_response = MagicMock()
    mock_response.json.return_value = [{"id": "sensor1"}]

    with patch("server.requests.get", return_value=mock_response), \
         patch("server.run_simulator") as mock_run_simulator, \
         patch("server.build_url", return_value="http://feeder:1234/sensor"), \
         patch("builtins.open", side_effect=patched_open):

        client = TestClient(app)
        response = client.post(
            "/run",
            json={"broker_ip": "127.0.0.1", "feeder_host": "feeder", "feeder_port": 1234},
        )

    assert response.status_code == 200
    assert "detail" in response.json()

    written = json.loads(sensors_file.read_text())
    assert written == [{"id": "sensor1"}]


# ---------------------------------------------------------------------------
# POST /configure — configure
# ---------------------------------------------------------------------------

def _configure_patched_open(tmp_path):
    """Return a patched open() that redirects config file writes to tmp_path."""
    from oedisi.types.common import DefaultFileNames
    import builtins
    real_open = builtins.open

    input_mapping_file = tmp_path / "input_mapping.json"
    static_inputs_file = tmp_path / "static_inputs.json"

    def patched_open(path, mode="r", *args, **kwargs):
        if path == DefaultFileNames.INPUT_MAPPING.value and "w" in mode:
            return real_open(str(input_mapping_file), mode, *args, **kwargs)
        if path == DefaultFileNames.STATIC_INPUTS.value and "w" in mode:
            return real_open(str(static_inputs_file), mode, *args, **kwargs)
        return real_open(path, mode, *args, **kwargs)

    return patched_open, input_mapping_file, static_inputs_file


def test_configure_writes_input_mapping_and_static_inputs(tmp_path):
    """T27 — valid ComponentStruct → writes both JSON files."""
    patched_open, input_mapping_file, static_inputs_file = _configure_patched_open(tmp_path)

    payload = {
        "component": {
            "name": "my_hub",
            "type": "HubPower",
            "parameters": {"max_itr": 5, "number_of_timesteps": 10},
        },
        "links": [
            {
                "source": "feeder",
                "source_port": "out",
                "target": "my_hub",
                "target_port": "in",
            }
        ],
    }

    with patch("builtins.open", side_effect=patched_open):
        client = TestClient(app)
        response = client.post("/configure", json=payload)

    assert response.status_code == 200
    assert "detail" in response.json()

    mapping = json.loads(input_mapping_file.read_text())
    assert mapping == {"in": "feeder/out"}

    static = json.loads(static_inputs_file.read_text())
    assert static["name"] == "my_hub"
    assert static["max_itr"] == 5


def test_configure_links_dict_construction(tmp_path):
    """T28 — links dict construction."""
    patched_open, input_mapping_file, _ = _configure_patched_open(tmp_path)

    payload = {
        "component": {
            "name": "hub",
            "type": "HubPower",
            "parameters": {},
        },
        "links": [
            {
                "source": "foo",
                "source_port": "out",
                "target": "hub",
                "target_port": "in",
            }
        ],
    }

    with patch("builtins.open", side_effect=patched_open):
        client = TestClient(app)
        client.post("/configure", json=payload)

    mapping = json.loads(input_mapping_file.read_text())
    assert mapping == {"in": "foo/out"}


def test_configure_empty_links(tmp_path):
    """T29 — empty links list → input_mapping.json is {}."""
    patched_open, input_mapping_file, _ = _configure_patched_open(tmp_path)

    payload = {
        "component": {
            "name": "hub",
            "type": "HubPower",
            "parameters": {"max_itr": 3},
        },
        "links": [],
    }

    with patch("builtins.open", side_effect=patched_open):
        client = TestClient(app)
        client.post("/configure", json=payload)

    mapping = json.loads(input_mapping_file.read_text())
    assert mapping == {}


def test_configure_empty_parameters(tmp_path):
    """T30 — empty parameters dict → static_inputs.json contains only name."""
    patched_open, _, static_inputs_file = _configure_patched_open(tmp_path)

    payload = {
        "component": {
            "name": "my_component",
            "type": "HubPower",
            "parameters": {},
        },
        "links": [],
    }

    with patch("builtins.open", side_effect=patched_open):
        client = TestClient(app)
        client.post("/configure", json=payload)

    static = json.loads(static_inputs_file.read_text())
    assert static == {"name": "my_component"}
