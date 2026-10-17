"""Native Qt desktop shell around the same offline, asynchronous application client."""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def ensure_backend(url, root):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    token_path = root / "access-token"

    def connected():
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=0.5) as response:
                data = json.load(response)
                return data.get("application") == "rocket-league-ai"
        except (OSError, ValueError):
            return False

    if not connected():
        parsed = urlsplit(url)
        if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.scheme != "http":
            raise RuntimeError("Automatic backend startup is local only. Start the specified backend explicitly.")
        fd = os.open(root / "backend.log", os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "a") as log:
            subprocess.Popen(
                [sys.executable, "-m", "rocket_league_ai", "backend", "--port", str(parsed.port or 80),
                 "--data-dir", str(root)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        deadline = time.monotonic() + 12
        while not connected():
            if time.monotonic() > deadline:
                raise RuntimeError("Backend did not become ready. Inspect the workspace's backend.log; check port and process ownership.")
            time.sleep(0.1)
    if not token_path.is_file():
        raise RuntimeError("Backend is running for another workspace. Select its data directory or a different port.")
    token = token_path.read_text().strip()
    request = urllib.request.Request(url + "/api/snapshot", headers={"Authorization": "Bearer " + token})
    try:
        urllib.request.urlopen(request, timeout=3).close()
    except urllib.error.HTTPError as error:
        raise RuntimeError("Backend rejected this workspace's key. Check --data-dir and --backend-url.") from error
    return token


def launch(args):
    # Import before starting any service so missing native dependencies yield an actionable error.
    from PySide6.QtCore import QStandardPaths, QTimer, QUrl
    from PySide6.QtGui import QColor
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow
    from shiboken6 import delete

    url = args.backend_url.rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.path not in {"", "/"}:
        raise RuntimeError("Use an HTTP(S) backend origin without credentials or a path.")
    application = QApplication(["Rocket League AI"])
    application.setApplicationName("Rocket League AI")
    token = ensure_backend(url, args.data_dir)
    window = QMainWindow()
    window.setWindowTitle("Rocket League AI — Control Room")
    window.resize(1440, 960)
    window.setMinimumSize(1000, 700)
    view = QWebEngineView(window)
    profile = QWebEngineProfile(window)  # Off-the-record: capability never written to a disk browser profile.

    class LocalPage(QWebEnginePage):
        def acceptNavigationRequest(self, target, navigation_type, is_main_frame):
            target_origin = (target.scheme(), target.host(), target.port())
            source = QUrl(url)
            allowed_origin = (source.scheme(), source.host(), source.port())
            return target_origin == allowed_origin or target.toString() == "about:blank"

        def createWindow(self, window_type):
            return None

    def save_download(download):
        # WebEngine does not automatically save Blob reports in a native shell.
        from pathlib import Path

        default = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation))
        filename, _ = QFileDialog.getSaveFileName(
            window, "Export measured workspace report", str(default / download.downloadFileName()),
            "JSON reports (*.json)",
        )
        if not filename:
            download.cancel()
            return
        target = Path(filename)
        download.setDownloadDirectory(str(target.parent))
        download.setDownloadFileName(target.name)
        download.accept()

    profile.downloadRequested.connect(save_download)
    page = LocalPage(profile, view)
    page.setBackgroundColor(QColor("#0b0f17"))
    page.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, False)
    page.settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, False)
    view.setPage(page)
    window.setCentralWidget(view)
    view.load(QUrl(url + "/#token=" + token))
    window.show()
    # There is deliberately NO aboutToQuit -> stop backend/worker connection.
    if args.smoke:
        deadline = time.monotonic() + 20
        timer = QTimer(window)

        routes = [
            ("dashboard", "Training, under control."), ("training", "Training"),
            ("monitoring", "Monitoring"), ("models", "Models"),
            ("configuration", "Configuration"), ("visualization", "Visualization"),
            ("dashboard", "Training, under control."),
        ]
        route_index = 0

        def result(snapshot):
            nonlocal route_index
            if not timer.isActive():
                return  # Qt may resolve a pending JS callback while destroying the page.
            if (isinstance(snapshot, dict) and snapshot.get("ready")
                    and (snapshot.get("route"), snapshot.get("heading")) == routes[route_index]):
                route_index += 1
                if route_index < len(routes):
                    page.runJavaScript(f"document.querySelector('[data-nav={routes[route_index][0]}]').click()")
                    return
                timer.stop()
                if args.screenshot:
                    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                    if not window.grab().save(str(args.screenshot)):
                        print("Desktop smoke failed: could not save screenshot", file=sys.stderr)
                        application.exit(1)
                        return
                print("Desktop smoke: six views loaded, backend connected; closing GUI only", flush=True)
                application.exit(0)
            elif time.monotonic() > deadline:
                timer.stop()
                print("Desktop smoke failed: six views did not connect/load in time", file=sys.stderr)
                application.exit(1)

        timer.timeout.connect(lambda: page.runJavaScript(
            "({ready: Boolean(window.__RLAI_READY__ && document.querySelector('#connection.online') "
            "&& document.querySelectorAll('[data-nav]').length === 6), "
            "route: document.querySelector('[data-nav][aria-current=page]')?.dataset.nav, "
            "heading: document.querySelector('main h1')?.textContent})", result,
        ))
        timer.start(200)
    code = application.exec()
    # Deterministic native destruction: the profile must outlive its page. Do not call
    # setPage(None) followed by deleteLater() on a potentially auto-deleted owned page.
    delete(page)
    delete(view)
    delete(profile)
    delete(window)
    return code
