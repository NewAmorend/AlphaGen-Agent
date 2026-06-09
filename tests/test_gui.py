from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from wq_agent.config import Settings
from wq_agent.gui.server import (
    CLEAR_SECRET_VALUE,
    CONFIG_FIELDS,
    MASKED_SECRET,
    MAX_JOB_LOG_LINES,
    EnvManager,
    Job,
    GuiState,
    JobManager,
    SAFE_ACTIONS,
    build_cli_command,
    _make_handler,
    _redact,
    STATIC_DIR,
)
from wq_agent.llm.factory import PROTOCOL_PROVIDER_OPTIONS
from http.server import ThreadingHTTPServer


def _start_test_server(tmp_path):
    state = GuiState(tmp_path, "127.0.0.1", 0)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    state.port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return state, server, thread


def _stop_test_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _open(request):
    return urllib.request.urlopen(request, timeout=5)


def test_env_snapshot_initializes_from_example_and_masks_secret(tmp_path):
    (tmp_path / ".env.example").write_text(
        "\n".join(
            [
                "LLM_API_KEY=secret-value",
                "LLM_MODEL=gpt-5.4",
                "WQ_USERNAME=alice",
                "WQ_PASSWORD=wq-secret",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manager = EnvManager(tmp_path)
    snapshot = manager.snapshot()
    values = {field["key"]: field for field in snapshot["fields"]}

    assert (tmp_path / ".env").exists()
    assert values["LLM_API_KEY"]["value"] == MASKED_SECRET
    assert values["LLM_API_KEY"]["has_value"] is True
    assert values["LLM_MODEL"]["value"] == "gpt-5.4"
    assert values["WQ_PASSWORD"]["value"] == MASKED_SECRET


def test_env_snapshot_uses_runtime_defaults_and_ignores_placeholder_secrets(tmp_path):
    (tmp_path / ".env.example").write_text(
        "\n".join(
            [
                "LLM_PROVIDER=openai_compatible",
                "LLM_API_KEY=your_openai_or_proxy_key",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manager = EnvManager(tmp_path)
    snapshot = manager.snapshot()
    values = {field["key"]: field for field in snapshot["fields"]}

    assert values["LLM_API_KEY"]["value"] == ""
    assert values["LLM_API_KEY"]["has_value"] is False
    assert values["LLM_BASE_URL"]["value"] == "https://api.openai.com/v1"
    assert values["LLM_MODEL"]["value"] == ""
    assert values["LLM_MAX_TOKENS"]["value"] == "32768"


def test_env_snapshot_treats_common_placeholder_secrets_as_empty(tmp_path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "LLM_API_KEY=change_me",
                "UNUSED_API_KEY=placeholder",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    values = {field["key"]: field for field in EnvManager(tmp_path).snapshot()["fields"]}

    assert values["LLM_API_KEY"]["has_value"] is False
    assert "UNUSED_API_KEY" not in values


def test_env_snapshot_and_settings_accept_utf8_bom_env_files(tmp_path):
    (tmp_path / ".env.example").write_text(
        "LLM_PROVIDER=anthropic\nLLM_MODEL=claude-custom\n",
        encoding="utf-8-sig",
    )

    manager = EnvManager(tmp_path)
    snapshot = manager.snapshot()
    values = {field["key"]: field for field in snapshot["fields"]}
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")

    assert values["LLM_PROVIDER"]["value"] == "anthropic"
    assert not env_text.startswith("\ufeff")

    settings = Settings(_env_file=str(tmp_path / ".env"))
    assert settings.LLM_PROVIDER == "anthropic"


def test_env_save_preserves_masked_secret_and_updates_plain_fields(tmp_path):
    (tmp_path / ".env").write_text(
        "LLM_API_KEY=secret-value\nLLM_MODEL=gpt-5.4\nWQ_USERNAME=alice\n",
        encoding="utf-8",
    )
    manager = EnvManager(tmp_path)

    manager.save(
        {
            "LLM_API_KEY": MASKED_SECRET,
            "LLM_MODEL": "gpt-5.5",
            "WQ_USERNAME": "bob",
        }
    )

    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "LLM_API_KEY=secret-value" in text
    assert "LLM_MODEL=gpt-5.5" in text
    assert "WQ_USERNAME=bob" in text


def test_env_save_can_clear_secret_explicitly(tmp_path):
    (tmp_path / ".env").write_text("LLM_API_KEY=secret-value\n", encoding="utf-8")

    snapshot = EnvManager(tmp_path).save({"LLM_API_KEY": CLEAR_SECRET_VALUE})
    values = {field["key"]: field for field in snapshot["fields"]}
    text = (tmp_path / ".env").read_text(encoding="utf-8")

    assert "LLM_API_KEY=" in text
    assert values["LLM_API_KEY"]["has_value"] is False


def test_env_save_deduplicates_keys_and_quotes_special_values(tmp_path):
    (tmp_path / ".env").write_text(
        "LLM_MODEL=old\nLLM_MODEL=older\nLLM_BASE_URL=https://api.openai.com/v1\n",
        encoding="utf-8",
    )
    manager = EnvManager(tmp_path)

    manager.save({"LLM_MODEL": "gpt custom", "LLM_BASE_URL": "https://proxy.example/v1"})

    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert text.count("LLM_MODEL=") == 1
    assert 'LLM_MODEL="gpt custom"' in text
    assert "LLM_BASE_URL=https://proxy.example/v1" in text


def test_env_save_rejects_invalid_select_number_and_accepts_custom_model(tmp_path):
    (tmp_path / ".env").write_text("LLM_PROVIDER=openai_compatible\nLLM_MODEL=\n", encoding="utf-8")
    manager = EnvManager(tmp_path)

    with pytest.raises(ValueError, match="LLM_WIRE_API"):
        manager.save({"LLM_WIRE_API": "bogus"})

    with pytest.raises(ValueError, match="LLM_MAX_TOKENS"):
        manager.save({"LLM_MAX_TOKENS": "0"})

    with pytest.raises(ValueError, match="WQ_MAX_CONCURRENT"):
        manager.save({"WQ_MAX_CONCURRENT": "999"})

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        manager.save({"LLM_API_KEY": "placeholder"})

    manager.save({"LLM_MODEL": "private-proxy-model"})
    assert "LLM_MODEL=private-proxy-model" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_config_model_options_stay_in_sync_with_factory_and_frontend():
    field_map = {field.key: field for field in CONFIG_FIELDS}

    assert field_map["LLM_PROVIDER"].options == PROTOCOL_PROVIDER_OPTIONS
    assert "openai" not in field_map["LLM_PROVIDER"].options
    assert "kimi" not in field_map["LLM_PROVIDER"].options
    assert "deepseek" not in field_map["LLM_PROVIDER"].options
    assert field_map["LLM_MODEL"].kind == "text"
    assert "OPENAI_MODEL" not in field_map
    assert "KIMI_MODEL" not in field_map
    assert "DEEPSEEK_MODEL" not in field_map

    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "GLOBAL_MODEL_OPTIONS" not in app_js
    assert "PROVIDER_MODEL_KEYS" not in app_js
    assert "PROVIDER_SECRET_KEYS" not in app_js


def test_build_cli_command_for_generate_and_backtest():
    generate = build_cli_command(
        "generate",
        {
            "strategy": "llm",
            "count": 5,
            "idea": "行业中性低换手",
            "no_backtest": True,
            "verbose": True,
        },
    )
    assert generate == [
        "generate",
        "--strategy",
        "llm",
        "--count",
        "5",
        "--idea",
        "行业中性低换手",
        "--no-backtest",
        "--verbose",
    ]

    backtest = build_cli_command(
        "backtest",
        {"mode": "ids", "ids": "1,2,3", "concurrent": 3},
    )
    assert backtest == ["backtest", "--concurrent", "3", "--ids", "1,2,3"]


@pytest.mark.parametrize(
    ("action", "payload", "expected"),
    [
        (
            "generate",
            {"strategy": "llm", "count": 2, "idea": "  ", "no_backtest": False},
            ["generate", "--strategy", "llm", "--count", "2"],
        ),
        (
            "generate",
            {"strategy": "factor_mining", "count": 1, "idea": "low turnover", "verbose": True},
            [
                "generate",
                "--strategy",
                "factor_mining",
                "--count",
                "1",
                "--idea",
                "low turnover",
                "--verbose",
            ],
        ),
        (
            "backtest",
            {"mode": "pending", "concurrent": 4},
            ["backtest", "--concurrent", "4", "--pending"],
        ),
        (
            "backtest",
            {"mode": "all", "concurrent": 2, "verbose": True},
            ["backtest", "--concurrent", "2", "--all", "--verbose"],
        ),
    ],
)
def test_gui_cli_command_regression_preserves_generate_backtest_contract(
    action,
    payload,
    expected,
):
    assert build_cli_command(action, payload) == expected


def test_submit_commands_are_not_gui_safe_actions():
    assert "submit" not in SAFE_ACTIONS
    assert "sync-submitted" not in SAFE_ACTIONS


def test_log_redaction_masks_known_secret_values():
    output = _redact(
        "Authorization: Bearer abc123 password=secret-value token=xyz",
        ["secret-value", "abc123"],
    )
    assert "abc123" not in output
    assert "secret-value" not in output
    assert "[REDACTED]" in output or MASKED_SECRET in output


def test_log_redaction_masks_json_and_openai_key_patterns():
    output = _redact(
        '{"api_key": "sk-proj-1234567890abcdef"} token="raw-token" '
        "LLM_API_KEY=sk-test-1234567890abcdef",
        [],
    )

    assert "sk-proj-1234567890abcdef" not in output
    assert "sk-test-1234567890abcdef" not in output
    assert "raw-token" not in output
    assert MASKED_SECRET in output or "[REDACTED]" in output


def test_static_assets_exist():
    assert (STATIC_DIR / "index.html").exists()
    assert (STATIC_DIR / "styles.css").exists()
    assert (STATIC_DIR / "app.js").exists()


def test_job_manager_runs_cli_help_to_completion(tmp_path):
    manager = JobManager(tmp_path, EnvManager(tmp_path))
    job = manager.start("help", ["--help"])

    deadline = time.time() + 10
    snapshot = manager.snapshot()["job"]
    while snapshot["status"] in {"pending", "running"} and time.time() < deadline:
        time.sleep(0.05)
        snapshot = manager.snapshot()["job"]

    assert job.id == snapshot["id"]
    assert snapshot["status"] == "completed"
    assert snapshot["returncode"] == 0
    assert any("Usage:" in line for line in snapshot["output"])


def test_job_snapshot_limits_log_lines():
    job = Job(id="job-1", action="generate", command=["python", "-m", "wq_agent.cli", "generate"])
    for i in range(MAX_JOB_LOG_LINES + 3):
        job.append_output(f"line-{i}")

    snapshot = job.snapshot()

    assert snapshot["output"][0].startswith("...仅保留最近")
    assert "line-0" not in snapshot["output"]
    assert f"line-{MAX_JOB_LOG_LINES + 2}" in snapshot["output"]
    assert len(snapshot["output"]) == MAX_JOB_LOG_LINES + 1


def test_job_manager_cancel_marks_job_cancelled(tmp_path):
    manager = JobManager(tmp_path, EnvManager(tmp_path))
    job = manager.start(
        "gui",
        ["gui", "--host", "127.0.0.1", "--port", "0", "--no-open-browser"],
    )

    deadline = time.time() + 10
    while time.time() < deadline:
        with manager._lock:
            process_started = manager._process is not None
        if process_started:
            break
        time.sleep(0.05)
    else:
        assert False, "GUI child process did not start"

    assert manager.cancel()["cancelled"] is True
    deadline = time.time() + 10
    snapshot = manager.snapshot()["job"]
    while snapshot["status"] in {"pending", "running", "cancelling"} and time.time() < deadline:
        time.sleep(0.05)
        snapshot = manager.snapshot()["job"]

    assert job.id == snapshot["id"]
    assert snapshot["status"] == "cancelled"
    assert any("停止" in line for line in snapshot["output"])


def test_http_post_requires_csrf_token_and_allows_valid_token(tmp_path):
    (tmp_path / ".env.example").write_text("LLM_MODEL=gpt-5.4\n", encoding="utf-8")
    state, server, thread = _start_test_server(tmp_path)
    try:
        url = f"http://127.0.0.1:{state.port}/api/config"
        body = json.dumps({"values": {"LLM_MODEL": "gpt-test"}}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(request, timeout=5)
            assert False, "POST without CSRF should fail"
        except urllib.error.HTTPError as exc:
            assert exc.code == 400

        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-WQ-Agent-CSRF": state.csrf_token,
                "Origin": f"http://127.0.0.1:{state.port}",
            },
        )
        with _open(request) as response:
            assert response.status == 200
        assert "LLM_MODEL=gpt-test" in (tmp_path / ".env").read_text(encoding="utf-8")
    finally:
        _stop_test_server(server, thread)


def test_http_get_api_requires_csrf_after_meta_and_sends_security_headers(tmp_path):
    (tmp_path / ".env.example").write_text("LLM_MODEL=gpt-5.4\n", encoding="utf-8")
    state, server, thread = _start_test_server(tmp_path)
    try:
        meta_request = urllib.request.Request(f"http://127.0.0.1:{state.port}/api/meta")
        with _open(meta_request) as response:
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert "default-src 'self'" in response.headers["Content-Security-Policy"]

        config_url = f"http://127.0.0.1:{state.port}/api/config"
        try:
            _open(urllib.request.Request(config_url))
            assert False, "GET /api/config without CSRF should fail"
        except urllib.error.HTTPError as exc:
            assert exc.code == 400

        request = urllib.request.Request(
            config_url,
            headers={
                "X-WQ-Agent-CSRF": state.csrf_token,
                "Origin": f"http://127.0.0.1:{state.port}",
            },
        )
        with _open(request) as response:
            assert response.status == 200
    finally:
        _stop_test_server(server, thread)


def test_http_rejects_invalid_host_and_origin(tmp_path):
    state, server, thread = _start_test_server(tmp_path)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", state.port, timeout=5)
        conn.request("GET", "/api/meta", headers={"Host": "evil.test"})
        response = conn.getresponse()
        assert response.status == 400
        conn.close()

        request = urllib.request.Request(
            f"http://127.0.0.1:{state.port}/api/config",
            headers={
                "X-WQ-Agent-CSRF": state.csrf_token,
                "Origin": "http://evil.test",
            },
        )
        try:
            _open(request)
            assert False, "GET with invalid Origin should fail"
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
    finally:
        _stop_test_server(server, thread)


def test_static_assets_reject_path_traversal_and_send_security_headers(tmp_path):
    state, server, thread = _start_test_server(tmp_path)
    try:
        index_request = urllib.request.Request(f"http://127.0.0.1:{state.port}/static/index.html")
        with _open(index_request) as response:
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"

        traversal = urllib.request.Request(
            f"http://127.0.0.1:{state.port}/static/%2e%2e%2fserver.py"
        )
        try:
            _open(traversal)
            assert False, "static path traversal should fail"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        _stop_test_server(server, thread)
