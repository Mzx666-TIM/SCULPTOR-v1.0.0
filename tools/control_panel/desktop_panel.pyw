"""SCULPTOR desktop control panel (native pywebview window, no server/browser).

Open by double-clicking tools/control_panel/open_control_panel.bat (or this .pyw).
The interface is identical to the web version; fetch("/api/...") is routed to
local Python by desktop_bridge.js.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import webview  # noqa: E402

import server as panel  # noqa: E402

SMOKE = bool(os.environ.get("FUEL_DESKTOP_SMOKE"))
SMOKE_LOG = Path(os.environ.get("FUEL_DESKTOP_SMOKE_LOG") or HERE / "desktop_smoke.log")


def smoke_log(msg: str) -> None:
    if not SMOKE:
        return
    try:
        with SMOKE_LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass


def build_desktop_html() -> Path:
    """Generate a desktop copy of index.html with relative asset paths."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    html = html.replace('href="/static/styles.css"', 'href="styles.css"')
    html = html.replace(
        'src="/static/app.js"',
        'src="desktop_bridge.js"></script>\n    <script src="app.js"',
    )
    out = STATIC / "desktop_index.html"
    out.write_text(html, encoding="utf-8")
    return out


class Api:
    """Exposed to JavaScript as window.pywebview.api.*."""

    def api_defaults(self) -> dict[str, object]:
        return panel.defaults_payload()

    def api_jobs(self) -> dict[str, object]:
        with panel.JOBS_LOCK:
            jobs = [job.as_dict() for job in panel.JOBS.values()]
        return {"jobs": jobs}

    def api_job(self, job_id: str) -> dict[str, object]:
        with panel.JOBS_LOCK:
            job = panel.JOBS.get(job_id)
        if job is None:
            raise ValueError("job not found")
        panel.refresh_job_results(job)
        return job.as_dict()

    def api_browse(self, path: str) -> dict[str, object]:
        return panel.browse_path(path)

    def api_preview(self, path: str) -> dict[str, object]:
        return panel.preview_file(path)

    def api_pick_folder(self, initial: str = "") -> dict[str, object]:
        return panel.pick_folder(initial or "")

    def api_build_command(self, payload: dict[str, object]) -> dict[str, object]:
        command, display, output_dir = panel.build_command(payload)
        return {
            "command": command,
            "display_command": display,
            "output_dir": output_dir,
        }

    def api_perturbation_inputs(self, payload: dict[str, object]) -> dict[str, object]:
        return panel.perturbation_inputs((payload or {}).get("output_dir"))

    def api_run(self, payload: dict[str, object]) -> dict[str, object]:
        command, display, output_dir = panel.build_command(payload)
        job = panel.create_job(
            mode=str(payload.get("mode", "")),
            command=command,
            display_command=display,
            output_dir=output_dir,
        )
        return job.as_dict()

    def api_resume(self, payload: dict[str, object]) -> dict[str, object]:
        panel.ensure_no_active_job()
        job_id = str(payload.get("job_id") or "")
        with panel.JOBS_LOCK:
            job = panel.JOBS.get(job_id)
        if job is None:
            raise ValueError("job not found")
        if job.status not in ("failed", "interrupted"):
            raise ValueError("only failed or interrupted jobs can be resumed")
        job.status = "queued"
        job.returncode = None
        job.ended_at = None
        panel.persist_job(job)
        threading.Thread(target=panel.run_job, args=(job,), daemon=True).start()
        return job.as_dict()

    def api_stop(self, payload: dict[str, object]) -> dict[str, object]:
        return panel.stop_job(str((payload or {}).get("job_id") or ""))


def main() -> int:
    smoke_log("main() begin")
    panel.JOB_ROOT.mkdir(parents=True, exist_ok=True)
    panel.load_persisted_jobs()
    threading.Thread(
        target=lambda: panel.recent_paths(panel.REPO_ROOT / "result"),
        daemon=True,
    ).start()

    html = build_desktop_html()
    smoke_log(f"html ready: {html}")
    api = Api()
    window = webview.create_window(
        "Fuel Molecule Design Control Panel (Desktop)",
        url=str(html),
        js_api=api,
        width=1320,
        height=900,
        min_size=(1024, 720),
        confirm_close=not SMOKE,
    )
    smoke_log("window created")

    if SMOKE:
        def _auto_close() -> None:
            time.sleep(10)
            try:
                smoke_log("auto-close: destroying window")
                window.destroy()
                smoke_log("auto-close: destroy() returned")
            except Exception:
                pass

        threading.Thread(target=_auto_close, daemon=True).start()

        def _verify() -> None:
            if not window.events.loaded.wait(12):
                smoke_log("page loaded event timeout")
                return
            time.sleep(2)
            try:
                js_setup = r"""
                window.__smokeResult = 'pending';
                (async () => {
                  try {
                    const title = document.title;
                    const controls = document.querySelectorAll('input, select, textarea, button').length;
                    let defaults = 'not-called';
                    try {
                      const resp = await window.fetch('/api/defaults');
                      const d = await resp.json();
                      defaults = resp.ok && d && typeof d === 'object'
                        ? 'keys=' + Object.keys(d).length
                        : 'bad:' + String(d);
                    } catch (e) { defaults = 'ERR:' + ((e && e.message) || e); }
                    const repo = document.getElementById('repoPath');
                    const status = document.getElementById('serverStatus');

                    let jobs = 'not-called';
                    try {
                      const resp = await window.fetch('/api/jobs');
                      const j = await resp.json();
                      jobs = resp.ok && j && Array.isArray(j.jobs) ? 'jobs=' + j.jobs.length : 'bad';
                    } catch (e) { jobs = 'ERR:' + ((e && e.message) || e); }

                    let browse = 'not-called';
                    let buildCmd = 'not-called';
                    try {
                      const d = await (await window.fetch('/api/defaults')).json();
                      const root = (d && d.repo_root) || '';
                      const resp = await window.fetch('/api/browse?path=' + encodeURIComponent(root));
                      const b = await resp.json();
                      browse = resp.ok && b && Array.isArray(b.entries) ? 'entries=' + b.entries.length : 'bad';

                      const payload = {
                        mode: 'symmetric',
                        params: {
                          checkpoint: d.defaults.checkpoint,
                          known_smiles: d.defaults.known_smiles,
                          output_dir: root + '/result/_desktop_smoke_test',
                        },
                      };
                      const resp2 = await window.fetch('/api/build-command', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                      });
                      const bc = await resp2.json();
                      buildCmd = resp2.ok && bc && bc.command && bc.command.length > 0
                        ? 'stages=' + bc.command.length
                        : 'bad';
                    } catch (e) {
                      if (browse === 'not-called') browse = 'ERR:' + ((e && e.message) || e);
                      if (buildCmd === 'not-called') buildCmd = 'ERR:' + ((e && e.message) || e);
                    }

                    window.__smokeResult = JSON.stringify({
                      title: title,
                      controls: controls,
                      bridgeLoaded: !!window.__desktopBridgeLoaded,
                      bridgePatched: !!window.__desktopBridgePatched,
                      defaults: defaults,
                      jobs: jobs,
                      browse: browse,
                      buildCmd: buildCmd,
                      repo: repo ? repo.textContent : 'missing',
                      status: status ? status.textContent : 'missing',
                    });
                  } catch (e) {
                    window.__smokeResult = 'FATAL:' + ((e && e.message) || e);
                  }
                })();
                """
                window.evaluate_js(js_setup)
                time.sleep(3)
                result = window.evaluate_js("window.__smokeResult || 'not-set'")
                smoke_log(f"verify result: {result}")
            except Exception as exc:  # pragma: no cover
                smoke_log(f"verify error: {exc!r}")

        threading.Thread(target=_verify, daemon=True).start()

    smoke_log("webview.start() calling")
    webview.start()
    smoke_log("webview.start() returned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
