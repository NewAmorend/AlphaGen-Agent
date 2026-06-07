from __future__ import annotations

import http.client
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
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
    serve_gui,
)
from wq_agent.workspace import WORKSPACE_ENV_VAR
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


def test_env_save_updates_wq_password_without_touching_api_keys(tmp_path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=openai-secret",
                "KIMI_API_KEY=kimi-secret",
                "DEEPSEEK_API_KEY=deepseek-secret",
                "WQ_PASSWORD=old-wq-secret",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    EnvManager(tmp_path).save({"WQ_PASSWORD": "new-wq-secret"})
    text = (tmp_path / ".env").read_text(encoding="utf-8")

    assert "WQ_PASSWORD=new-wq-secret" in text
    assert "OPENAI_API_KEY=openai-secret" in text
    assert "KIMI_API_KEY=kimi-secret" in text
    assert "DEEPSEEK_API_KEY=deepseek-secret" in text


def test_env_save_updates_api_key_without_touching_wq_password(tmp_path):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=old-openai-secret\nWQ_PASSWORD=wq-secret\n",
        encoding="utf-8",
    )

    EnvManager(tmp_path).save({"OPENAI_API_KEY": "new-openai-secret"})
    text = (tmp_path / ".env").read_text(encoding="utf-8")

    assert "OPENAI_API_KEY=new-openai-secret" in text
    assert "WQ_PASSWORD=wq-secret" in text


def test_env_save_rejects_matching_wq_password_and_api_key_updates(tmp_path):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=old-openai-secret\nWQ_PASSWORD=old-wq-secret\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="自动填充污染"):
        EnvManager(tmp_path).save(
            {
                "OPENAI_API_KEY": "same-secret-value",
                "WQ_PASSWORD": "same-secret-value",
            }
        )

    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=old-openai-secret" in text
    assert "WQ_PASSWORD=old-wq-secret" in text


def test_env_save_rejects_api_key_matching_current_wq_password(tmp_path):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=old-openai-secret\nWQ_PASSWORD=current-wq-secret\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="自动填充污染"):
        EnvManager(tmp_path).save({"OPENAI_API_KEY": "current-wq-secret"})

    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=old-openai-secret" in text
    assert "WQ_PASSWORD=current-wq-secret" in text


def test_env_save_rejects_wq_password_matching_current_api_key(tmp_path):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=current-openai-secret\nWQ_PASSWORD=old-wq-secret\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="自动填充污染"):
        EnvManager(tmp_path).save({"WQ_PASSWORD": "current-openai-secret"})

    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=current-openai-secret" in text
    assert "WQ_PASSWORD=old-wq-secret" in text


def test_env_save_clear_one_secret_only_affects_that_key(tmp_path):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=openai-secret\nWQ_PASSWORD=wq-secret\n",
        encoding="utf-8",
    )

    EnvManager(tmp_path).save({"OPENAI_API_KEY": CLEAR_SECRET_VALUE})
    text = (tmp_path / ".env").read_text(encoding="utf-8")

    assert "OPENAI_API_KEY=" in text
    assert "WQ_PASSWORD=wq-secret" in text


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


def test_frontend_secrets_require_explicit_editing_before_save():
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'input.dataset.secret = "true"' in app_js
    assert 'input.dataset.secretEditing = "false"' in app_js
    assert 'input.disabled = true' in app_js
    assert 'isSecret && input.dataset.secretEditing !== "true"' in app_js
    assert "return;" in app_js
    assert "autocompleteForField" in app_js
    assert "wq-agent-${cssToken(field.key)}" in app_js


def test_frontend_collect_config_values_secret_payload_behavior(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for frontend behavior coverage")

    script = tmp_path / "config_payload_test.js"
    script.write_text(
        r"""
const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

let currentInputs = [];
const context = {
  document: {
    addEventListener() {},
    getElementById() { return null; },
    querySelectorAll(selector) {
      assert.strictEqual(selector, "[data-config-key]");
      return currentInputs;
    },
  },
  window: { setTimeout() {} },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), context, { filename: "app.js" });

function collect() {
  return JSON.parse(JSON.stringify(context.collectConfigValues()));
}

function secret(key, options = {}) {
  return {
    dataset: {
      configKey: key,
      secret: "true",
      secretEditing: options.editing || "false",
      clearSecret: options.clear || "false",
    },
    type: "password",
    value: options.value || "",
  };
}

function plain(key, value) {
  return { dataset: { configKey: key }, type: "text", value };
}

function checkbox(key, checked) {
  return { dataset: { configKey: key }, type: "checkbox", checked };
}

currentInputs = [
  secret("OPENAI_API_KEY", { value: "autofilled-wq-password" }),
  secret("WQ_PASSWORD", { editing: "true", value: "new-wq-password" }),
  plain("OPENAI_MODEL", "gpt-5.5"),
];
assert.deepStrictEqual(collect(), {
  WQ_PASSWORD: "new-wq-password",
  OPENAI_MODEL: "gpt-5.5",
});

currentInputs = [
  secret("OPENAI_API_KEY", { editing: "true", value: "new-openai-key" }),
  secret("WQ_PASSWORD", { value: "autofilled-wq-password" }),
];
assert.deepStrictEqual(collect(), {
  OPENAI_API_KEY: "new-openai-key",
});

currentInputs = [
  secret("OPENAI_API_KEY", { clear: "true", value: "should-not-send" }),
  secret("WQ_PASSWORD", { value: "should-not-send" }),
];
assert.deepStrictEqual(collect(), {
  OPENAI_API_KEY: "__clear_secret__",
});

currentInputs = [
  secret("OPENAI_API_KEY", { editing: "true", value: "" }),
  checkbox("OPENAI_STORE", true),
];
assert.deepStrictEqual(collect(), {
  OPENAI_STORE: true,
});
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [node, str(script), str(STATIC_DIR / "app.js")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout


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


def test_job_manager_passes_workspace_env_to_child_process(tmp_path, monkeypatch):
    captured = {}

    class FakeProcess:
        stdout = ["done\n"]

        def wait(self):
            return 0

    def fake_popen(command, *, cwd, env, stdout, stderr, text, encoding, errors):
        captured.update(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "stdout": stdout,
                "stderr": stderr,
                "text": text,
                "encoding": encoding,
                "errors": errors,
            }
        )
        return FakeProcess()

    monkeypatch.setattr("wq_agent.gui.server.subprocess.Popen", fake_popen)
    manager = JobManager(tmp_path, EnvManager(tmp_path))
    manager.start("status", ["status"])

    deadline = time.time() + 5
    snapshot = manager.snapshot()["job"]
    while snapshot["status"] in {"pending", "running"} and time.time() < deadline:
        time.sleep(0.01)
        snapshot = manager.snapshot()["job"]

    assert snapshot["status"] == "completed"
    assert captured["cwd"] == tmp_path
    assert captured["env"][WORKSPACE_ENV_VAR] == str(tmp_path)


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


def test_http_config_and_wiki_use_workspace_root(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text(
        "OPENAI_MODEL=gpt-workspace\nWIKI_DIR=./wiki\nWIKI_AUTO_RECORD_DIR=./private_wiki\n",
        encoding="utf-8",
    )
    (workspace / "wiki").mkdir()
    (workspace / "wiki" / "index.md").write_text("# Workspace Wiki", encoding="utf-8")
    (workspace / "private_wiki").mkdir()
    (workspace / "private_wiki" / "secret-alpha.md").write_text(
        "# Private Alpha",
        encoding="utf-8",
    )
    dist_root = workspace / "dist" / "wq-agent"
    dist_root.mkdir(parents=True)
    (dist_root / ".env").write_text("OPENAI_MODEL=gpt-dist\n", encoding="utf-8")

    state, server, thread = _start_test_server(workspace)
    try:
        meta_request = urllib.request.Request(f"http://127.0.0.1:{state.port}/api/meta")
        with _open(meta_request) as response:
            token = json.loads(response.read().decode("utf-8"))["csrf_token"]

        headers = {
            "X-WQ-Agent-CSRF": token,
            "Origin": f"http://127.0.0.1:{state.port}",
        }
        with _open(
            urllib.request.Request(
                f"http://127.0.0.1:{state.port}/api/config",
                headers=headers,
            )
        ) as response:
            config = json.loads(response.read().decode("utf-8"))
        values = {field["key"]: field["value"] for field in config["fields"]}

        with _open(
            urllib.request.Request(
                f"http://127.0.0.1:{state.port}/api/wiki/tree",
                headers=headers,
            )
        ) as response:
            tree = json.loads(response.read().decode("utf-8"))

        assert config["env_path"] == str(workspace / ".env")
        assert values["OPENAI_MODEL"] == "gpt-workspace"
        assert tree["roots"]["private"]["path"] == str(workspace / "private_wiki")
        assert tree["roots"]["private"]["file_count"] == 1
        assert tree["roots"]["private"]["tree"]["children"][0]["name"] == "secret-alpha.md"
    finally:
        _stop_test_server(server, thread)


def test_serve_gui_resolves_workspace_from_frozen_dist(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    exe_dir = workspace / "dist" / "wq-agent"
    exe_dir.mkdir(parents=True)
    exe_path = exe_dir / "wq-agent.exe"
    exe_path.write_text("", encoding="utf-8")
    states = []

    class FakeServer:
        def __init__(self, address, handler):
            self.address = address
            self.handler = handler

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    def fake_make_handler(state):
        states.append(state)
        return object()

    monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_path))
    monkeypatch.setattr("wq_agent.gui.server._make_handler", fake_make_handler)
    monkeypatch.setattr("wq_agent.gui.server.ThreadingHTTPServer", FakeServer)
    old_cwd = Path.cwd()

    try:
        serve_gui(open_browser=False)
    finally:
        os.chdir(old_cwd)

    assert states[0].root == workspace.resolve()
    assert states[0].env.env_path == workspace.resolve() / ".env"


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
