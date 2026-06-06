from __future__ import annotations

import http.client
import io
import json
import sys
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
    build_subprocess_command,
    _make_handler,
    _redact,
    STATIC_DIR,
)
from wq_agent.gui.wiki_files import (
    UploadedFile,
    build_wiki_tree,
    import_uploaded_files,
    read_wiki_file,
)
from wq_agent.llm.factory import GLOBAL_MODEL_OPTIONS
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


def _multipart_body(
    fields: dict[str, str],
    files: list[tuple[str, str, bytes, str]],
    *,
    boundary: str = "----WQAgentTestBoundary",
) -> tuple[bytes, str]:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, filename, content, content_type in files:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def test_env_snapshot_initializes_from_example_and_masks_secret(tmp_path):
    (tmp_path / ".env.example").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=secret-value",
                "OPENAI_MODEL=gpt-5.4",
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
    assert values["OPENAI_API_KEY"]["value"] == MASKED_SECRET
    assert values["OPENAI_API_KEY"]["has_value"] is True
    assert values["OPENAI_MODEL"]["value"] == "gpt-5.4"
    assert values["WQ_PASSWORD"]["value"] == MASKED_SECRET


def test_env_snapshot_uses_runtime_defaults_and_ignores_placeholder_secrets(tmp_path):
    (tmp_path / ".env.example").write_text(
        "\n".join(
            [
                "LLM_PROVIDER=deepseek",
                "DEEPSEEK_API_KEY=your_deepseek_key",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manager = EnvManager(tmp_path)
    snapshot = manager.snapshot()
    values = {field["key"]: field for field in snapshot["fields"]}

    assert values["DEEPSEEK_API_KEY"]["value"] == ""
    assert values["DEEPSEEK_API_KEY"]["has_value"] is False
    assert values["DEEPSEEK_BASE_URL"]["value"] == "https://api.deepseek.com/v1/chat/completions"
    assert values["DEEPSEEK_MODEL"]["value"] == "deepseek-chat"
    assert values["LLM_MAX_TOKENS"]["value"] == "32768"


def test_env_snapshot_treats_common_placeholder_secrets_as_empty(tmp_path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=change_me",
                "KIMI_API_KEY=placeholder",
                "DEEPSEEK_API_KEY=todo-fill-me",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    values = {field["key"]: field for field in EnvManager(tmp_path).snapshot()["fields"]}

    assert values["OPENAI_API_KEY"]["has_value"] is False
    assert values["KIMI_API_KEY"]["has_value"] is False
    assert values["DEEPSEEK_API_KEY"]["has_value"] is False


def test_env_snapshot_and_settings_accept_utf8_bom_env_files(tmp_path):
    (tmp_path / ".env.example").write_text(
        "LLM_PROVIDER=deepseek\nOPENAI_MODEL=gpt-5.4\n",
        encoding="utf-8-sig",
    )

    manager = EnvManager(tmp_path)
    snapshot = manager.snapshot()
    values = {field["key"]: field for field in snapshot["fields"]}
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")

    assert values["LLM_PROVIDER"]["value"] == "deepseek"
    assert not env_text.startswith("\ufeff")

    settings = Settings(_env_file=str(tmp_path / ".env"))
    assert settings.LLM_PROVIDER == "deepseek"


def test_env_save_preserves_masked_secret_and_updates_plain_fields(tmp_path):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=secret-value\nOPENAI_MODEL=gpt-5.4\nWQ_USERNAME=alice\n",
        encoding="utf-8",
    )
    manager = EnvManager(tmp_path)

    manager.save(
        {
            "OPENAI_API_KEY": MASKED_SECRET,
            "OPENAI_MODEL": "gpt-5.5",
            "WQ_USERNAME": "bob",
        }
    )

    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=secret-value" in text
    assert "OPENAI_MODEL=gpt-5.5" in text
    assert "WQ_USERNAME=bob" in text


def test_env_save_can_clear_secret_explicitly(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=secret-value\n", encoding="utf-8")

    snapshot = EnvManager(tmp_path).save({"OPENAI_API_KEY": CLEAR_SECRET_VALUE})
    values = {field["key"]: field for field in snapshot["fields"]}
    text = (tmp_path / ".env").read_text(encoding="utf-8")

    assert "OPENAI_API_KEY=" in text
    assert values["OPENAI_API_KEY"]["has_value"] is False


def test_env_save_deduplicates_keys_and_quotes_special_values(tmp_path):
    (tmp_path / ".env").write_text(
        "OPENAI_MODEL=old\nOPENAI_MODEL=older\nOPENAI_BASE_URL=https://api.openai.com/v1\n",
        encoding="utf-8",
    )
    manager = EnvManager(tmp_path)

    manager.save({"OPENAI_MODEL": "gpt custom", "OPENAI_BASE_URL": "https://proxy.example/v1"})

    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert text.count("OPENAI_MODEL=") == 1
    assert 'OPENAI_MODEL="gpt custom"' in text
    assert "OPENAI_BASE_URL=https://proxy.example/v1" in text


def test_env_save_rejects_invalid_select_number_and_provider_model(tmp_path):
    (tmp_path / ".env").write_text("LLM_PROVIDER=deepseek\nLLM_MODEL=\n", encoding="utf-8")
    manager = EnvManager(tmp_path)

    with pytest.raises(ValueError, match="OPENAI_WIRE_API"):
        manager.save({"OPENAI_WIRE_API": "bogus"})

    with pytest.raises(ValueError, match="LLM_MAX_TOKENS"):
        manager.save({"LLM_MAX_TOKENS": "0"})

    with pytest.raises(ValueError, match="WQ_MAX_CONCURRENT"):
        manager.save({"WQ_MAX_CONCURRENT": "999"})

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        manager.save({"OPENAI_API_KEY": "placeholder"})

    with pytest.raises(ValueError, match="LLM_MODEL"):
        manager.save({"LLM_MODEL": "gpt-5.4"})

    manager.save({"LLM_MODEL": "deepseek-reasoner"})
    assert "LLM_MODEL=deepseek-reasoner" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_env_save_clears_incompatible_hidden_global_model_on_provider_switch(tmp_path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "LLM_PROVIDER=openai",
                "LLM_MODEL=gpt-5.4",
                "KIMI_MODEL=kimi-k2.6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    EnvManager(tmp_path).save({"LLM_PROVIDER": "kimi"})
    text = (tmp_path / ".env").read_text(encoding="utf-8")

    assert "LLM_PROVIDER=kimi" in text
    assert "LLM_MODEL=" in text
    assert "LLM_MODEL=gpt-5.4" not in text


def test_env_save_clears_compatible_hidden_global_model_when_provider_model_is_saved(tmp_path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "LLM_PROVIDER=kimi",
                "LLM_MODEL=kimi-k2.6",
                "KIMI_MODEL=kimi-k2.6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    EnvManager(tmp_path).save({"KIMI_MODEL": "kimi-custom"})
    text = (tmp_path / ".env").read_text(encoding="utf-8")

    assert "LLM_MODEL=" in text
    assert "KIMI_MODEL=kimi-custom" in text
    assert "LLM_MODEL=kimi-k2.6" not in text


def test_env_save_preserves_hidden_provider_values_on_provider_switch(tmp_path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "LLM_PROVIDER=openai",
                "OPENAI_API_KEY=openai-secret",
                "OPENAI_MODEL=gpt-5.4",
                "KIMI_API_KEY=kimi-secret",
                "KIMI_MODEL=kimi-k2.6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    EnvManager(tmp_path).save(
        {
            "LLM_PROVIDER": "kimi",
            "LLM_MAX_TOKENS": "32768",
            "KIMI_MODEL": "kimi-k2.6",
        }
    )
    text = (tmp_path / ".env").read_text(encoding="utf-8")

    assert "LLM_PROVIDER=kimi" in text
    assert "OPENAI_API_KEY=openai-secret" in text
    assert "OPENAI_MODEL=gpt-5.4" in text
    assert "KIMI_API_KEY=kimi-secret" in text
    assert "KIMI_MODEL=kimi-k2.6" in text


def test_config_model_options_stay_in_sync_with_factory_and_frontend():
    field_map = {field.key: field for field in CONFIG_FIELDS}
    expected_global = {""}
    for models in GLOBAL_MODEL_OPTIONS.values():
        expected_global.update(models)

    assert set(field_map["LLM_MODEL"].options) == expected_global
    assert set(field_map["LLM_PROVIDER"].options) == set(GLOBAL_MODEL_OPTIONS)

    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    for provider, models in GLOBAL_MODEL_OPTIONS.items():
        assert provider in app_js
        for model in models:
            assert model in app_js


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
        "OPENAI_API_KEY=sk-test-1234567890abcdef",
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


def test_wiki_file_import_builds_private_upload_markdown_and_tree(tmp_path):
    (tmp_path / ".env").write_text(
        "WIKI_DIR=./wiki\nWIKI_AUTO_RECORD_DIR=./private_wiki\n",
        encoding="utf-8",
    )
    (tmp_path / "wiki" / "concepts").mkdir(parents=True)
    (tmp_path / "wiki" / "concepts" / "momentum.md").write_text(
        "# Momentum\n\npublic note",
        encoding="utf-8",
    )

    result = import_uploaded_files(
        tmp_path,
        [UploadedFile(filename="../alpha idea.txt", content="低换手 alpha".encode("utf-8"))],
    )

    uploaded = result["uploaded"][0]
    assert uploaded["path"].startswith("uploads/")
    assert uploaded["path"].endswith("-alpha-idea.md")
    assert result["tree"]["roots"]["public"]["file_count"] == 1
    assert result["tree"]["roots"]["private"]["file_count"] == 1

    saved = tmp_path / "private_wiki" / uploaded["path"]
    text = saved.read_text(encoding="utf-8")
    assert "original_filename: alpha idea.txt" in text
    assert "# alpha idea" in text
    assert "低换手 alpha" in text

    payload = read_wiki_file(tmp_path, "private", uploaded["path"])
    assert payload["name"] == saved.name
    assert "低换手 alpha" in payload["content"]


def test_wiki_file_import_rejects_unsupported_and_unsafe_paths(tmp_path):
    (tmp_path / ".env").write_text(
        "WIKI_DIR=./wiki\nWIKI_AUTO_RECORD_DIR=./private_wiki\n",
        encoding="utf-8",
    )
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "note.md").write_text("# Note", encoding="utf-8")

    with pytest.raises(ValueError, match="not supported"):
        import_uploaded_files(
            tmp_path,
            [UploadedFile(filename="data.csv", content=b"a,b\n1,2")],
        )

    with pytest.raises(ValueError, match="Invalid wiki path"):
        read_wiki_file(tmp_path, "public", "../.env")

    with pytest.raises(ValueError, match="Only Markdown"):
        read_wiki_file(tmp_path, "public", "note.txt")


def test_wiki_docx_import_extracts_text_and_enforces_text_limit(tmp_path, monkeypatch):
    from docx import Document

    (tmp_path / ".env").write_text(
        "WIKI_DIR=./wiki\nWIKI_AUTO_RECORD_DIR=./private_wiki\n",
        encoding="utf-8",
    )
    doc = Document()
    doc.add_paragraph("short alpha memo")
    buffer = io.BytesIO()
    doc.save(buffer)

    result = import_uploaded_files(
        tmp_path,
        [UploadedFile(filename="memo.docx", content=buffer.getvalue())],
    )
    uploaded = result["uploaded"][0]
    saved = tmp_path / "private_wiki" / uploaded["path"]

    assert uploaded["source_type"] == "docx"
    assert "short alpha memo" in saved.read_text(encoding="utf-8")

    monkeypatch.setattr("wq_agent.gui.wiki_files.MAX_EXTRACTED_CHARS", 4)
    with pytest.raises(ValueError, match="extracted text exceeds"):
        import_uploaded_files(
            tmp_path,
            [UploadedFile(filename="too-long.docx", content=buffer.getvalue())],
        )


def test_wiki_roots_must_stay_under_workspace(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside-wiki"
    (tmp_path / ".env").write_text(
        f"WIKI_DIR={outside}\nWIKI_AUTO_RECORD_DIR=./private_wiki\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="under the workspace"):
        build_wiki_tree(tmp_path)


def test_wiki_tree_reports_public_and_private_roots(tmp_path):
    (tmp_path / ".env").write_text(
        "WIKI_DIR=./wiki\nWIKI_AUTO_RECORD_DIR=./private_wiki\n",
        encoding="utf-8",
    )
    (tmp_path / "wiki" / "operators").mkdir(parents=True)
    (tmp_path / "wiki" / "operators" / "rank.md").write_text("# Rank", encoding="utf-8")

    tree = build_wiki_tree(tmp_path)

    assert tree["roots"]["public"]["label"] == "公开知识库"
    assert tree["roots"]["public"]["file_count"] == 1
    assert tree["roots"]["private"]["label"] == "私有知识库"
    assert tree["roots"]["private"]["exists"] is False


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


def test_job_manager_builds_subprocess_command_for_runtime_modes(monkeypatch):
    monkeypatch.setattr(sys, "executable", r"C:\dist\wq-agent\wq-agent.exe")
    monkeypatch.delattr(sys, "frozen", raising=False)

    assert build_subprocess_command(["status"]) == [
        r"C:\dist\wq-agent\wq-agent.exe",
        "-m",
        "wq_agent.cli",
        "status",
    ]

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert build_subprocess_command(["gui", "--no-open-browser"]) == [
        r"C:\dist\wq-agent\wq-agent.exe",
        "gui",
        "--no-open-browser",
    ]


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
    (tmp_path / ".env.example").write_text("OPENAI_MODEL=gpt-5.4\n", encoding="utf-8")
    state, server, thread = _start_test_server(tmp_path)
    try:
        url = f"http://127.0.0.1:{state.port}/api/config"
        body = json.dumps({"values": {"OPENAI_MODEL": "gpt-test"}}).encode("utf-8")
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
        assert "OPENAI_MODEL=gpt-test" in (tmp_path / ".env").read_text(encoding="utf-8")
    finally:
        _stop_test_server(server, thread)


def test_http_get_api_requires_csrf_after_meta_and_sends_security_headers(tmp_path):
    (tmp_path / ".env.example").write_text("OPENAI_MODEL=gpt-5.4\n", encoding="utf-8")
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


def test_http_wiki_upload_accepts_multipart_with_csrf_and_updates_tree(tmp_path):
    (tmp_path / ".env").write_text(
        "WIKI_DIR=./wiki\nWIKI_AUTO_RECORD_DIR=./private_wiki\n",
        encoding="utf-8",
    )
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "index.md").write_text("# Wiki", encoding="utf-8")
    state, server, thread = _start_test_server(tmp_path)
    try:
        body, content_type = _multipart_body(
            {"root": "private"},
            [
                (
                    "files",
                    "research-note.md",
                    "# Research\n\nalpha memo".encode("utf-8"),
                    "text/markdown",
                )
            ],
        )
        request = urllib.request.Request(
            f"http://127.0.0.1:{state.port}/api/wiki/upload",
            data=body,
            method="POST",
            headers={
                "Content-Type": content_type,
                "X-WQ-Agent-CSRF": state.csrf_token,
                "Origin": f"http://127.0.0.1:{state.port}",
            },
        )

        with _open(request) as response:
            assert response.status == 200
            payload = json.loads(response.read().decode("utf-8"))

        uploaded = payload["uploaded"][0]
        assert uploaded["original_name"] == "research-note.md"
        assert uploaded["path"].startswith("uploads/")
        assert payload["tree"]["roots"]["private"]["file_count"] == 1
        assert (tmp_path / "private_wiki" / uploaded["path"]).exists()
    finally:
        _stop_test_server(server, thread)


def test_http_wiki_upload_rejects_public_root_and_malformed_multipart(tmp_path):
    (tmp_path / ".env").write_text(
        "WIKI_DIR=./wiki\nWIKI_AUTO_RECORD_DIR=./private_wiki\n",
        encoding="utf-8",
    )
    (tmp_path / "wiki").mkdir()
    state, server, thread = _start_test_server(tmp_path)
    try:
        body, content_type = _multipart_body(
            {"root": "public"},
            [("files", "secret.md", b"# Secret", "text/markdown")],
        )
        request = urllib.request.Request(
            f"http://127.0.0.1:{state.port}/api/wiki/upload",
            data=body,
            method="POST",
            headers={
                "Content-Type": content_type,
                "X-WQ-Agent-CSRF": state.csrf_token,
                "Origin": f"http://127.0.0.1:{state.port}",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as public_error:
            _open(request)
        assert public_error.value.code == 400
        assert not (tmp_path / "wiki" / "uploads").exists()

        malformed = urllib.request.Request(
            f"http://127.0.0.1:{state.port}/api/wiki/upload",
            data=b"not a multipart body",
            method="POST",
            headers={
                "Content-Type": "multipart/form-data",
                "X-WQ-Agent-CSRF": state.csrf_token,
                "Origin": f"http://127.0.0.1:{state.port}",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as multipart_error:
            _open(malformed)
        assert multipart_error.value.code == 400
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
