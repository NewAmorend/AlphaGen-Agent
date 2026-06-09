from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import uuid
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from dotenv import dotenv_values

from ..config import Settings
from ..db import Database
from ..llm.factory import PROTOCOL_PROVIDER_OPTIONS
from ..llm.security import is_real_secret


MASKED_SECRET = "********"
CLEAR_SECRET_VALUE = "__clear_secret__"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost"}
MAX_JOB_LOG_LINES = 500
CANCEL_TIMEOUT_SECONDS = 3
STATIC_DIR = Path(__file__).with_name("static")
SECRET_KEYS = {
    "LLM_API_KEY",
    "WQ_PASSWORD",
    "KIMI_API_KEY",
    "EMBEDDING_API_KEY",
}
NUMBER_MINIMUMS = {
    "LLM_MAX_TOKENS": 1,
    "WQ_DELAY": 0,
    "WQ_MAX_CONCURRENT": 1,
}
NUMBER_MAXIMUMS = {
    "LLM_MAX_TOKENS": 200000,
    "WQ_DELAY": 10,
    "WQ_MAX_CONCURRENT": 20,
}


@dataclass(frozen=True)
class ConfigField:
    key: str
    label: str
    section: str
    secret: bool = False
    kind: str = "text"
    options: tuple[str, ...] = ()
    allow_custom: bool = False


CONFIG_FIELDS: tuple[ConfigField, ...] = (
    ConfigField(
        "LLM_PROVIDER",
        "LLM Provider",
        "模型",
        kind="select",
        options=tuple(PROTOCOL_PROVIDER_OPTIONS),
    ),
    ConfigField("LLM_BASE_URL", "LLM API 地址", "模型"),
    ConfigField("LLM_API_KEY", "LLM API 密钥", "模型", secret=True, kind="password"),
    ConfigField("LLM_MODEL", "LLM 模型", "模型"),
    ConfigField("LLM_MAX_TOKENS", "最大输出 Token", "模型", kind="number"),
    ConfigField("LLM_WIRE_API", "Wire API", "模型", kind="select", options=("auto", "responses", "chat_completions")),
    ConfigField(
        "LLM_REASONING_EFFORT",
        "Reasoning Effort",
        "模型",
        kind="select",
        options=("", "none", "minimal", "low", "medium", "high", "xhigh"),
    ),
    ConfigField("LLM_STORE", "响应存储", "模型", kind="boolean"),
    ConfigField("LLM_ALLOW_INSECURE_HTTP", "允许远程 HTTP", "模型", kind="boolean"),
    ConfigField(
        "LLM_CHAT_TOKEN_PARAM",
        "Chat Token 参数",
        "模型",
        kind="select",
        options=("max_tokens", "max_completion_tokens"),
    ),
    ConfigField("LLM_CHAT_REASONING_EFFORT", "Chat Reasoning 参数", "模型", kind="boolean"),
    ConfigField("ANTHROPIC_VERSION", "Anthropic Version", "模型"),
    ConfigField("WQ_USERNAME", "WQ 用户名", "WorldQuant"),
    ConfigField("WQ_PASSWORD", "WQ 密码", "WorldQuant", secret=True, kind="password"),
    ConfigField("WQ_REGION", "Region", "WorldQuant"),
    ConfigField("WQ_UNIVERSE", "Universe", "WorldQuant"),
    ConfigField("WQ_DELAY", "Delay", "WorldQuant", kind="number"),
    ConfigField("WQ_NEUTRALIZATION", "Neutralization", "WorldQuant"),
    ConfigField("WQ_MAX_CONCURRENT", "回测并发", "WorldQuant", kind="number"),
    ConfigField("DB_PATH", "数据库路径", "本地"),
    ConfigField("EMBEDDING_PROVIDER", "Embedding Provider", "本地"),
)
CONFIG_FIELD_MAP = {field.key: field for field in CONFIG_FIELDS}

SAFE_ACTIONS = {"generate", "run", "backtest", "refine"}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ENV_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


class EnvManager:
    def __init__(self, root: Path):
        self.root = root
        self.env_path = root / ".env"
        self.example_path = root / ".env.example"

    def snapshot(self) -> dict[str, Any]:
        self.ensure_env()
        values = self._read_values()
        fields = []
        for field_def in CONFIG_FIELDS:
            raw_value = values.get(field_def.key)
            if field_def.secret:
                has_value = is_real_secret(raw_value)
                value = MASKED_SECRET if has_value else ""
            else:
                value = raw_value if raw_value is not None else _settings_default(field_def.key)
                has_value = bool(value)
            fields.append(
                {
                    "key": field_def.key,
                    "label": field_def.label,
                    "section": field_def.section,
                    "kind": field_def.kind,
                    "options": list(field_def.options),
                    "allow_custom": field_def.allow_custom,
                    "secret": field_def.secret,
                    "has_value": has_value,
                    "value": value,
                }
            )
        return {
            "env_exists": self.env_path.exists(),
            "env_path": str(self.env_path),
            "fields": fields,
        }

    def save(self, updates: dict[str, Any]) -> dict[str, Any]:
        self.ensure_env()
        lines = self.env_path.read_text(encoding="utf-8-sig").splitlines()
        seen_keys: set[str] = set()

        normalized: dict[str, str] = {}
        raw_normalized: dict[str, str] = {}
        for key, value in updates.items():
            field_def = CONFIG_FIELD_MAP.get(key)
            if not field_def:
                continue
            text = self._normalize_value(value)
            if field_def.secret and text in {"", MASKED_SECRET}:
                continue
            if field_def.secret and text == CLEAR_SECRET_VALUE:
                normalized[key] = ""
                raw_normalized[key] = ""
                continue
            text = _validate_config_value(field_def, text)
            raw_normalized[key] = text
            normalized[key] = self._quote_value(text)

        _validate_config_updates(self._read_values(), raw_normalized)

        updated_lines: list[str] = []
        for line in lines:
            match = ENV_KEY_RE.match(_clean_env_line(line).strip())
            if not match:
                updated_lines.append(line)
                continue
            key = match.group(1)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if key in normalized:
                updated_lines.append(f"{key}={normalized.pop(key)}")
            else:
                updated_lines.append(line)

        if normalized:
            if updated_lines and updated_lines[-1].strip():
                updated_lines.append("")
            for key, value in normalized.items():
                if key not in seen_keys:
                    updated_lines.append(f"{key}={value}")

        tmp_path = self.env_path.with_suffix(self.env_path.suffix + ".tmp")
        tmp_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
        tmp_path.replace(self.env_path)
        return self.snapshot()

    def ensure_env(self) -> None:
        if self.env_path.exists():
            return
        if self.example_path.exists():
            self.env_path.write_text(
                self.example_path.read_text(encoding="utf-8-sig"), encoding="utf-8"
            )
        else:
            self.env_path.write_text("", encoding="utf-8")

    def _read_values(self) -> dict[str, str]:
        if not self.env_path.exists():
            return {}
        values = dotenv_values(self.env_path, encoding="utf-8-sig")
        return {_clean_env_key(key): "" if value is None else value for key, value in values.items()}

    @staticmethod
    def _normalize_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).replace("\r", " ").replace("\n", " ").strip()

    @staticmethod
    def _quote_value(value: str) -> str:
        if value == "":
            return ""
        if re.search(r"\s|#|=|\"|'", value):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return value

    def secret_values(self) -> list[str]:
        values = self._read_values()
        secrets_to_hide = []
        for key in SECRET_KEYS:
            value = values.get(key, "")
            if is_real_secret(value):
                secrets_to_hide.append(value)
        return secrets_to_hide


def build_cli_command(action: str, payload: dict[str, Any]) -> list[str]:
    if action not in SAFE_ACTIONS:
        raise ValueError(f"Unsupported GUI action: {action}")

    if action == "generate":
        command = ["generate", "--strategy", _strategy(payload), "--count", _int_str(payload, "count", 18)]
        idea = str(payload.get("idea", "")).strip()
        if idea:
            command += ["--idea", idea]
        if _bool(payload.get("no_backtest")):
            command.append("--no-backtest")
        if _bool(payload.get("verbose")):
            command.append("--verbose")
        return command

    if action == "run":
        command = [
            "run",
            "--strategy",
            _strategy(payload),
            "--count",
            _int_str(payload, "count", 18),
            "--batches",
            _int_str(payload, "batches", 1),
            "--interval",
            _int_str(payload, "interval", 60, minimum=0),
        ]
        idea = str(payload.get("idea", "")).strip()
        if idea:
            command += ["--idea", idea]
        if _bool(payload.get("verbose")):
            command.append("--verbose")
        return command

    if action == "backtest":
        mode = str(payload.get("mode", "pending")).strip().lower()
        command = ["backtest", "--concurrent", _int_str(payload, "concurrent", 5, minimum=1, maximum=20)]
        if mode == "pending":
            command.append("--pending")
        elif mode == "all":
            command.append("--all")
        elif mode == "ids":
            ids = str(payload.get("ids", "")).strip()
            if not re.fullmatch(r"\d+(,\s*\d+)*", ids):
                raise ValueError("ids must be comma-separated numbers")
            command += ["--ids", ids]
        else:
            raise ValueError("backtest mode must be pending, all, or ids")
        if _bool(payload.get("verbose")):
            command.append("--verbose")
        return command

    command = ["refine", "--count", _int_str(payload, "count", 10)]
    base_id = str(payload.get("base_id", "")).strip()
    if base_id:
        if not base_id.isdigit():
            raise ValueError("base_id must be a number")
        command += ["--base-id", base_id]
    if _bool(payload.get("no_backtest")):
        command.append("--no-backtest")
    if _bool(payload.get("verbose")):
        command.append("--verbose")
    return command


@dataclass
class Job:
    id: str
    action: str
    command: list[str]
    status: str = "pending"
    started_at: str | None = None
    ended_at: str | None = None
    returncode: int | None = None
    output: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_JOB_LOG_LINES))
    output_truncated: bool = False

    def append_output(self, line: str) -> None:
        if len(self.output) == self.output.maxlen:
            self.output_truncated = True
        self.output.append(line)

    def snapshot(self, secrets_to_hide: list[str] | None = None) -> dict[str, Any]:
        secrets_to_hide = secrets_to_hide or []
        output = [_redact(line, secrets_to_hide) for line in self.output]
        if self.output_truncated:
            output.insert(0, f"...仅保留最近 {MAX_JOB_LOG_LINES} 行日志...")
        return {
            "id": self.id,
            "action": self.action,
            "command": [self.command[0], "-m", "wq_agent.cli", self.action],
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "returncode": self.returncode,
            "output": output,
        }


class JobManager:
    def __init__(self, root: Path, env: EnvManager):
        self.root = root
        self.env = env
        self._lock = threading.Lock()
        self._job: Job | None = None
        self._process: subprocess.Popen[str] | None = None

    def start(self, action: str, cli_args: list[str]) -> Job:
        with self._lock:
            if self._job and self._job.status in {"pending", "running", "cancelling"}:
                raise RuntimeError("已有任务正在运行，请等待完成后再启动新任务")
            command = [sys.executable, "-m", "wq_agent.cli", *cli_args]
            job = Job(id=str(uuid.uuid4()), action=action, command=command)
            self._job = job
            thread = threading.Thread(target=self._run, args=(job,), daemon=True)
            thread.start()
            return job

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            job = self._job
        if process and process.poll() is None:
            if job:
                job.status = "cancelling"
                job.append_output("任务已请求停止。")
            process.terminate()
            try:
                process.wait(timeout=CANCEL_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=CANCEL_TIMEOUT_SECONDS)
                if job:
                    job.append_output("任务未及时停止，已强制结束。")
            return {"cancelled": True}
        return {"cancelled": False}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if not self._job:
                return {"job": None}
            return {"job": self._job.snapshot(self.env.secret_values())}

    def _run(self, job: Job) -> None:
        job.status = "running"
        job.started_at = _now()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("NO_COLOR", "1")
        try:
            process = subprocess.Popen(
                job.command,
                cwd=self.root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            with self._lock:
                self._process = process
            assert process.stdout is not None
            for line in process.stdout:
                job.append_output(_redact(ANSI_RE.sub("", line.rstrip()), self.env.secret_values()))
            job.returncode = process.wait()
            if job.status == "cancelling":
                job.status = "cancelled"
            else:
                job.status = "completed" if job.returncode == 0 else "failed"
        except Exception as exc:  # pragma: no cover - defensive guard for UI resilience
            job.status = "failed"
            job.append_output(f"GUI 任务执行失败：{exc}")
        finally:
            job.ended_at = _now()
            with self._lock:
                self._process = None


class GuiState:
    def __init__(self, root: Path, host: str, port: int):
        self.root = root
        self.host = host
        self.port = port
        self.csrf_token = secrets.token_urlsafe(32)
        self.env = EnvManager(root)
        self.jobs = JobManager(root, self.env)

    @property
    def allowed_hosts(self) -> set[str]:
        return {f"{self.host}:{self.port}", f"localhost:{self.port}", f"127.0.0.1:{self.port}"}

    @property
    def origin(self) -> str:
        return f"http://{self.host}:{self.port}"


def serve_gui(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    if host not in LOOPBACK_HOSTS:
        raise ValueError("GUI 只能绑定 127.0.0.1 / localhost")
    state = GuiState(Path.cwd(), host=host, port=port)
    handler = _make_handler(state)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"wq-agent GUI 已启动：{url}")
    print("按 Ctrl+C 停止。")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止 GUI...")
    finally:
        httpd.server_close()


def _make_handler(state: GuiState):
    class GuiRequestHandler(BaseHTTPRequestHandler):
        server_version = "WQAgentGUI/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            try:
                if self.path.startswith("/api/"):
                    self._validate_read_request(allow_missing_token=self.path == "/api/meta")
                if self.path == "/" or self.path == "/index.html":
                    self._send_file(STATIC_DIR / "index.html")
                elif self.path.startswith("/static/"):
                    self._send_static_asset()
                elif self.path == "/api/config":
                    self._send_json(state.env.snapshot())
                elif self.path == "/api/meta":
                    self._send_json(
                        {
                            "csrf_token": state.csrf_token,
                            "workspace": str(state.root),
                            "origin": state.origin,
                        }
                    )
                elif self.path == "/api/results":
                    self._send_json(_load_results(state.root))
                elif self.path == "/api/job":
                    self._send_json(state.jobs.snapshot())
                else:
                    self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:
            try:
                self._validate_write_request()
                payload = self._read_json()
                if self.path == "/api/config":
                    self._send_json(state.env.save(payload.get("values", payload)))
                elif self.path == "/api/run":
                    action = str(payload.get("action", "")).strip()
                    cli_args = build_cli_command(action, payload)
                    job = state.jobs.start(action, cli_args)
                    self._send_json({"job": job.snapshot(state.env.secret_values())})
                elif self.path == "/api/job/cancel":
                    self._send_json(state.jobs.cancel())
                else:
                    self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            data = self.rfile.read(length)
            return json.loads(data.decode("utf-8"))

        def _validate_write_request(self) -> None:
            self._validate_host_and_origin()
            content_type = self.headers.get("Content-Type", "")
            if "application/json" not in content_type.lower():
                raise ValueError("POST requests must use application/json")
            token = self.headers.get("X-WQ-Agent-CSRF", "")
            if token != state.csrf_token:
                raise ValueError("Invalid CSRF token")

        def _validate_read_request(self, *, allow_missing_token: bool = False) -> None:
            self._validate_host_and_origin()
            token = self.headers.get("X-WQ-Agent-CSRF", "")
            if not allow_missing_token and token != state.csrf_token:
                raise ValueError("Invalid CSRF token")

        def _validate_host_and_origin(self) -> None:
            host = self.headers.get("Host", "")
            if host not in state.allowed_hosts:
                raise ValueError("Invalid Host header")
            origin = self.headers.get("Origin") or self.headers.get("Referer")
            if origin:
                parsed = urlparse(origin)
                if parsed.netloc not in state.allowed_hosts:
                    raise ValueError("Invalid request origin")

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(data)

        def _send_static_asset(self) -> None:
            parsed_path = urlparse(self.path).path
            raw_name = unquote(parsed_path.removeprefix("/static/"))
            if "/" in raw_name or "\\" in raw_name or raw_name in {"", ".", ".."}:
                self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_file(STATIC_DIR / raw_name)

        def _send_file(self, path: Path) -> None:
            if not path.exists():
                self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
                return
            content = path.read_bytes()
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", _content_type(path))
            self.send_header("Content-Length", str(len(content)))
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(content)

        def _send_security_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; base-uri 'none'; frame-ancestors 'none'",
            )

    return GuiRequestHandler


def _load_results(root: Path) -> dict[str, Any]:
    async def _run() -> dict[str, Any]:
        settings = Settings(_env_file=str(root / ".env") if (root / ".env").exists() else None)
        db_path = Path(settings.DB_PATH)
        if not db_path.is_absolute():
            db_path = root / db_path
        db = Database(str(db_path))
        await db.connect()
        try:
            stats = await db.get_stats()
            alphas = await db.list_alphas(limit=50)
            recent = []
            for alpha in alphas:
                result = await db.get_backtest_result(alpha.id or 0)
                recent.append(
                    {
                        "id": alpha.id,
                        "expression": alpha.expression,
                        "strategy": alpha.strategy.value,
                        "status": alpha.status.value,
                        "llm_model": alpha.llm_model,
                        "created_at": alpha.created_at.isoformat(),
                        "fitness": result.fitness if result else None,
                        "sharpe": result.sharpe if result else None,
                        "turnover": result.turnover if result else None,
                        "grade": result.grade.value if result and result.grade else None,
                        "wq_alpha_id": result.wq_alpha_id if result else None,
                    }
                )
            submittable = await db.list_submittable_alphas(limit=50)
            return {"stats": stats, "recent": recent, "submittable": submittable}
        finally:
            await db.close()

    return asyncio.run(_run())


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".js":
        return "text/javascript; charset=utf-8"
    return "application/octet-stream"


def _strategy(payload: dict[str, Any]) -> str:
    strategy = str(payload.get("strategy", "llm")).strip()
    if strategy not in {"llm", "template", "factor_mining"}:
        raise ValueError("strategy must be llm, template, or factor_mining")
    return strategy


def _int_str(
    payload: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: int = 1000,
) -> str:
    value = int(payload.get(key, default) or default)
    if value < minimum or value > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return str(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _validate_config_value(field_def: ConfigField, text: str) -> str:
    if field_def.secret:
        if not is_real_secret(text):
            raise ValueError(f"{field_def.key} must be a real secret or left blank")
        return text

    if field_def.kind == "select":
        if field_def.allow_custom and text:
            return text
        if text not in field_def.options:
            allowed = ", ".join(option or "<blank>" for option in field_def.options)
            raise ValueError(f"{field_def.key} must be one of: {allowed}")
        return text

    if field_def.kind == "boolean":
        normalized = text.lower()
        if normalized in {"1", "true", "yes", "on"}:
            return "true"
        if normalized in {"0", "false", "no", "off"}:
            return "false"
        raise ValueError(f"{field_def.key} must be true or false")

    if field_def.kind == "number":
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{field_def.key} must be a number") from exc
        minimum = NUMBER_MINIMUMS.get(field_def.key, 0)
        if value < minimum:
            raise ValueError(f"{field_def.key} must be {minimum} or greater")
        maximum = NUMBER_MAXIMUMS.get(field_def.key)
        if maximum is not None and value > maximum:
            raise ValueError(f"{field_def.key} must be {maximum} or less")
        return str(value)

    return text


def _validate_config_updates(current_values: dict[str, str], updates: dict[str, str]) -> None:
    return


def _clean_env_key(key: str) -> str:
    return key.lstrip("\ufeff")


def _clean_env_line(line: str) -> str:
    return line.lstrip("\ufeff")


def _settings_default(key: str) -> str:
    field = Settings.model_fields.get(key)
    if not field:
        return ""
    value = field.default
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _redact(text: str, secrets_to_hide: list[str]) -> str:
    redacted = text
    for value in secrets_to_hide:
        if value:
            redacted = redacted.replace(value, MASKED_SECRET)
    redacted = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,\"']+",
        r"\1[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)((?:api[_-]?key|token|secret|password)\s*[:=]\s*)[^\s,\"']+",
        r"\1[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)([\"'](?:api[_-]?key|token|secret|password)[\"']\s*:\s*[\"'])[^\"']+([\"'])",
        rf"\1{MASKED_SECRET}\2",
        redacted,
    )
    redacted = re.sub(
        r"(?i)((?:api[_-]?key|token|secret|password)\s*[:=]\s*[\"'])[^\"']+([\"'])",
        rf"\1{MASKED_SECRET}\2",
        redacted,
    )
    redacted = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", MASKED_SECRET, redacted)
    return redacted
