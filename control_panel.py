"""Local browser control panel for the quant review server."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
QUANT_BAT = ROOT / "run_quant.bat"
CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 8764
REVIEW_HOST = "127.0.0.1"
REVIEW_PORT = 8765
CONTROL_URL = f"http://{CONTROL_HOST}:{CONTROL_PORT}/"
REVIEW_URL = f"http://{REVIEW_HOST}:{REVIEW_PORT}/"
CONTROL_TOKEN = "quant-local-static-control-v1-7f36c982"
STATIC_CONTROL_HTML = ROOT / "quant_control.html"
REVIEW_CONTROL_HEADER = "X-Quant-Control"
REVIEW_CONTROL_VALUE = "quant-control-panel-v1"

_process_lock = threading.Lock()
_quant_process: subprocess.Popen[bytes] | None = None


CONTROL_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>量化复盘控制台</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f6fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #657084;
      --line: #dbe2ec;
      --primary: #1769e0;
      --primary-dark: #1054b8;
      --danger: #c92a2a;
      --good: #087f5b;
      --shadow: 0 18px 48px rgba(29, 50, 84, .12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      background:
        radial-gradient(circle at top left, rgba(23, 105, 224, .14), transparent 34rem),
        var(--bg);
      color: var(--text);
      font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    }
    main {
      width: min(720px, 100%);
      padding: 30px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    h1 { margin: 0 0 8px; font-size: 28px; }
    .intro { margin: 0; color: var(--muted); line-height: 1.7; }
    .status-card {
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 26px 0 20px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #f8fafc;
    }
    .dot {
      width: 12px;
      height: 12px;
      flex: 0 0 auto;
      border-radius: 50%;
      background: #98a2b3;
      box-shadow: 0 0 0 5px rgba(152, 162, 179, .14);
    }
    .dot.running {
      background: var(--good);
      box-shadow: 0 0 0 5px rgba(8, 127, 91, .14);
    }
    .status-label { font-weight: 800; }
    .status-detail { margin-top: 3px; color: var(--muted); font-size: 13px; }
    .actions {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    button, .link-button {
      min-height: 46px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 0;
      border-radius: 10px;
      padding: 0 16px;
      font: inherit;
      font-weight: 800;
      text-decoration: none;
      cursor: pointer;
      transition: transform .15s ease, background .15s ease, opacity .15s ease;
    }
    button:hover, .link-button:hover { transform: translateY(-1px); }
    button:disabled { cursor: wait; opacity: .58; transform: none; }
    .primary { background: var(--primary); color: #fff; }
    .primary:hover { background: var(--primary-dark); }
    .secondary { background: #e9f1fd; color: var(--primary-dark); }
    .danger { background: #fff0f0; color: var(--danger); }
    .open { border: 1px solid var(--line); background: #fff; color: var(--text); }
    .message {
      min-height: 24px;
      margin: 18px 0 0;
      color: var(--muted);
      line-height: 1.6;
    }
    .message.error { color: var(--danger); font-weight: 700; }
    .hint {
      margin: 20px 0 0;
      padding-top: 18px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.7;
    }
    code { color: var(--text); }
    @media (max-width: 600px) {
      main { padding: 22px; }
      h1 { font-size: 23px; }
      .actions { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <h1>量化复盘控制台</h1>
    <p class="intro">在浏览器里启动或停止复盘服务，也可以直接打开复盘页面。</p>

    <section class="status-card" aria-live="polite">
      <span id="statusDot" class="dot"></span>
      <div>
        <div id="statusLabel" class="status-label">正在检查服务状态...</div>
        <div id="statusDetail" class="status-detail">复盘地址：__REVIEW_URL__</div>
      </div>
    </section>

    <div class="actions">
      <button class="primary" data-action onclick="startReview(true)">启动并打开复盘</button>
      <button class="secondary" data-action onclick="startReview(false)">仅启动复盘</button>
      <button class="danger" data-action onclick="stopReview()">停止复盘</button>
      <a class="link-button open" href="__REVIEW_URL__" target="quant-review">直接打开复盘</a>
    </div>

    <p id="message" class="message">控制页会自动刷新运行状态。</p>
    <p class="hint">
      控制页只监听本机 <code>127.0.0.1</code>。关闭复盘不会关闭本控制页，
      因此之后仍可从这里重新启动。
    </p>
  </main>

  <script>
    const controlToken = __CONTROL_TOKEN__;
    const reviewUrl = __REVIEW_URL_JSON__;
    let busy = false;

    function setBusy(value) {
      busy = value;
      document.querySelectorAll("[data-action]").forEach(button => {
        button.disabled = value;
      });
    }

    function renderStatus(data) {
      const running = Boolean(data.running);
      document.getElementById("statusDot").classList.toggle("running", running);
      document.getElementById("statusLabel").textContent = running
        ? "复盘服务正在运行"
        : "复盘服务已停止";
      document.getElementById("statusDetail").textContent = running
        ? `可以访问：${reviewUrl}`
        : "点击“启动并打开复盘”即可开始";
    }

    function showMessage(text, isError = false) {
      const element = document.getElementById("message");
      element.textContent = text;
      element.classList.toggle("error", isError);
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        ...options,
        headers: {
          "X-Control-Token": controlToken,
          ...(options.headers || {})
        }
      });
      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "操作失败");
      }
      return data;
    }

    async function refreshStatus() {
      if (busy) return;
      try {
        renderStatus(await api("/api/status"));
      } catch (error) {
        showMessage(`状态检查失败：${error.message}`, true);
      }
    }

    async function startReview(openAfterStart) {
      const reviewWindow = openAfterStart ? window.open("about:blank", "quant-review") : null;
      setBusy(true);
      showMessage("正在启动复盘服务...");
      try {
        const data = await api("/api/start", {method: "POST"});
        renderStatus(data);
        showMessage(data.message);
        if (reviewWindow) reviewWindow.location.replace(reviewUrl);
      } catch (error) {
        if (reviewWindow) reviewWindow.close();
        showMessage(`启动失败：${error.message}`, true);
      } finally {
        setBusy(false);
      }
    }

    async function stopReview() {
      setBusy(true);
      showMessage("正在停止复盘服务...");
      try {
        const data = await api("/api/stop", {method: "POST"});
        renderStatus(data);
        showMessage(data.message);
      } catch (error) {
        showMessage(`停止失败：${error.message}`, true);
      } finally {
        setBusy(false);
      }
    }

    refreshStatus();
    window.setInterval(refreshStatus, 2000);
  </script>
</body>
</html>
"""


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | int(
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    )


def _review_running(timeout: float = 0.45) -> bool:
    try:
        with urllib.request.urlopen(f"{REVIEW_URL}api/health", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and payload.get("service") == "quant-review":
                return True
    except (OSError, ValueError, urllib.error.URLError):
        pass

    # Compatibility with a review process started before /api/health was added.
    try:
        with urllib.request.urlopen(REVIEW_URL, timeout=timeout) as response:
            body = response.read(256_000)
            return response.status == 200 and b"/app-layout.css" in body
    except (OSError, urllib.error.URLError):
        pass
    return False


def _wait_for_review(running: bool, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _review_running() is running:
            return True
        time.sleep(0.2)
    return _review_running() is running


def _find_review_pid() -> int | None:
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            timeout=4,
            creationflags=_creation_flags(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local_address = parts[1]
        if not local_address.endswith(f":{REVIEW_PORT}"):
            continue
        try:
            pid = int(parts[-1])
        except ValueError:
            continue
        if pid > 0 and pid != os.getpid():
            return pid
    return None


def _kill_process_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=8,
        creationflags=_creation_flags(),
        check=False,
    )


def start_review() -> dict[str, Any]:
    global _quant_process

    if _review_running():
        return {"ok": True, "running": True, "message": "复盘服务已经在运行。"}
    if not QUANT_BAT.exists():
        raise RuntimeError(f"找不到启动文件：{QUANT_BAT}")

    with _process_lock:
        if _quant_process is None or _quant_process.poll() is not None:
            command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(QUANT_BAT)]
            _quant_process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=_creation_flags(),
            )

    if not _wait_for_review(True, 12):
        exit_code = _quant_process.poll() if _quant_process is not None else None
        detail = f"，退出码 {exit_code}" if exit_code is not None else ""
        raise RuntimeError(f"复盘服务未能在 12 秒内启动{detail}")

    return {"ok": True, "running": True, "message": "复盘服务已启动。"}


def stop_review() -> dict[str, Any]:
    global _quant_process

    if not _review_running():
        return {"ok": True, "running": False, "message": "复盘服务已经停止。"}

    shutdown_request = urllib.request.Request(
        f"{REVIEW_URL}api/shutdown",
        data=b"{}",
        headers={
            "Content-Type": "application/json",
            REVIEW_CONTROL_HEADER: REVIEW_CONTROL_VALUE,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(shutdown_request, timeout=3):
            pass
    except (OSError, urllib.error.URLError):
        pass

    if not _wait_for_review(False, 5):
        pid = _find_review_pid()
        if pid is not None:
            _kill_process_tree(pid)

    with _process_lock:
        if _quant_process is not None and _quant_process.poll() is None:
            _kill_process_tree(_quant_process.pid)
        _quant_process = None

    if not _wait_for_review(False, 3):
        raise RuntimeError("复盘服务仍在运行，请检查 8765 端口")
    return {"ok": True, "running": False, "message": "复盘服务已停止。"}


class ControlHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            html = STATIC_CONTROL_HTML.read_text(encoding="utf-8")
            self._send(200, html, "text/html; charset=utf-8")
            return
        if path == "/api/status":
            if not self._authorized():
                return
            self._send_json({"ok": True, "running": _review_running()})
            return
        self._send(404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if not self._authorized():
            return
        path = urlparse(self.path).path
        try:
            if path == "/api/start":
                self._send_json(start_review())
            elif path == "/api/stop":
                self._send_json(stop_review())
            else:
                self._send_json({"ok": False, "error": "unknown endpoint"}, 404)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, 500)

    def _authorized(self) -> bool:
        if self.client_address[0] not in {"127.0.0.1", "::1"}:
            self._send_json({"ok": False, "error": "local access only"}, 403)
            return False
        if self.headers.get("X-Control-Token") != CONTROL_TOKEN:
            self._send_json({"ok": False, "error": "invalid control token"}, 403)
            return False
        return True

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False),
            "application/json; charset=utf-8",
        )

    def _send(self, status: int, body: str, content_type: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(encoded)

    def _send_cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin in {None, "null", CONTROL_URL.rstrip("/")}:
            self.send_header("Access-Control-Allow-Origin", origin or "null")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "X-Control-Token, Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Vary", "Origin")

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    try:
        server = ThreadingHTTPServer((CONTROL_HOST, CONTROL_PORT), ControlHandler)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 10048:
            return
        raise

    print(f"Quant control panel: {CONTROL_URL}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
