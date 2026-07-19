"""
live_viewer.py
==============

A real-time view of a run *while it is happening*. The runner starts a small
background web server; open the printed URL in a browser and the page polls for
the latest turn and refreshes itself, so you can watch each decision land: the
game state, the prompt, the model's raw response, the action, the think time,
and a running success tally.

This is separate from ``viewer.py`` (which builds a static replay you scrub
through *after* a run). Use this one to confirm an experiment is working live.

    live = LiveViewer(run_dir, "my_exp", "collect wood")
    url = live.start()          # -> http://127.0.0.1:8000
    live.update({...})          # call once per turn
    live.set_complete()         # run finished
    live.stop()

Nothing here is AI-facing; the server only ever reports state to your browser.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG = logging.getLogger("crafter_experiment.live")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
POLL_MS = 700  # how often the browser asks for the latest turn


class LiveViewer:
    """Serves a self-refreshing page showing the current turn of a run."""

    def __init__(
        self,
        run_dir: Path,
        experiment_name: str,
        objective_label: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ):
        self._run_dir = Path(run_dir)
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._frame_png: bytes | None = None  # latest frame, served at /frame.png
        # The single snapshot every browser poll reads.
        self._state: dict = {
            "experiment": experiment_name,
            "objective": objective_label,
            "status": "starting",
            "model": None,
        }

    # =========================================================================
    #  Lifecycle
    # =========================================================================
    def start(self, open_browser: bool = True) -> str:
        """Bind the server, serve in a daemon thread, and open a browser tab.

        Returns the URL. ``open_browser=False`` skips the auto-open (e.g. on a
        headless machine); the URL is always logged either way.
        """
        self._run_dir.mkdir(parents=True, exist_ok=True)
        handler = partial(_Handler, self, directory=str(self._run_dir))

        port = self._bind(handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        url = f"http://{self._host}:{port}"
        LOG.info("=" * 60)
        LOG.info("LIVE VIEW: open %s in your browser", url)
        LOG.info("=" * 60)
        if open_browser:
            # Open a tab once the server thread is up. Non-fatal if it fails
            # (headless box, no default browser) - the URL is logged above.
            try:
                threading.Timer(0.6, webbrowser.open, args=(url,)).start()
            except Exception:  # pragma: no cover - environment dependent
                pass
        return url

    def _bind(self, handler) -> int:
        """Try the requested port, then fall back to an OS-chosen free one."""
        for candidate in (self._port, 0):
            try:
                self._server = ThreadingHTTPServer((self._host, candidate), handler)
                return self._server.server_address[1]
            except OSError:
                continue
        raise OSError("could not bind the live-view server to any port")

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def serve_until_interrupt(self) -> None:
        """Keep the final state viewable until the user presses Ctrl-C."""
        if self._server is None:
            return
        LOG.info("Live view still serving the final state - press Ctrl-C to exit.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    # =========================================================================
    #  State publishing (called by the runner)
    # =========================================================================
    def update(self, snapshot: dict) -> None:
        """Replace the live snapshot with the newest turn (thread-safe)."""
        with self._lock:
            self._state = {**self._state, **snapshot, "status": "running"}

    def set_complete(self) -> None:
        with self._lock:
            self._state["status"] = "complete"

    def set_frame(self, png_bytes: bytes) -> None:
        """Publish the latest rendered frame (served at /frame.png)."""
        with self._lock:
            self._frame_png = png_bytes

    def frame_png(self) -> bytes | None:
        with self._lock:
            return self._frame_png

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)


# =============================================================================
#  Request handler
# =============================================================================
class _Handler(SimpleHTTPRequestHandler):
    """Serves the page at '/', JSON at '/state', and frame PNGs statically."""

    def __init__(self, viewer: LiveViewer, *args, **kwargs):
        self._viewer = viewer
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 (name mandated by base class)
        if self.path == "/" or self.path.startswith("/index"):
            self._send_html(_PAGE)
        elif self.path.startswith("/state"):
            self._send_json(self._viewer.snapshot())
        elif self.path.startswith("/frame.png"):
            self._send_frame(self._viewer.frame_png())
        else:
            # Anything else (e.g. a video under /videos/) is served from the
            # run directory by the parent class.
            super().do_GET()

    def _send_frame(self, png: bytes | None) -> None:
        if not png:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(png)

    def _send_html(self, body: str) -> None:
        raw = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_json(self, obj: dict) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args) -> None:  # silence per-request console spam
        return


# =============================================================================
#  The page (polls /state and re-renders)
# =============================================================================
_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Crafter &mdash; live</title>
<style>
  :root {
    --bg:#12141a; --panel:#1b1e27; --panel-2:#232733; --line:#2f3444;
    --ink:#e6e8ef; --muted:#8b90a3; --accent:#e0a44b; --ok:#4bd39a; --bad:#e0655f;
    --mono:"SFMono-Regular",ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace;
    --sans:"Inter",system-ui,-apple-system,"Segoe UI",sans-serif;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans); font-size:14px; }
  header { padding:16px 24px; border-bottom:1px solid var(--line); display:flex;
           align-items:center; gap:16px; flex-wrap:wrap; }
  header h1 { font-size:14px; letter-spacing:.14em; text-transform:uppercase; margin:0; font-weight:600; }
  .live { display:flex; align-items:center; gap:7px; font-family:var(--mono); font-size:12px; color:var(--muted); }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--ok); box-shadow:0 0 0 0 rgba(75,211,154,.6); }
  .dot.on { animation:pulse 1.4s infinite; }
  .dot.done { background:var(--muted); animation:none; }
  @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(75,211,154,.5);} 70%{box-shadow:0 0 0 8px rgba(75,211,154,0);} 100%{box-shadow:0 0 0 0 rgba(75,211,154,0);} }
  .goal { color:var(--accent); font-family:var(--mono); font-size:13px; }
  .goal b { color:var(--ink); }
  main { display:grid; grid-template-columns:minmax(300px,440px) 1fr; gap:1px; background:var(--line);
         min-height:calc(100vh - 58px); }
  .stage, .side { background:var(--bg); padding:20px 24px; }
  .stage { display:flex; flex-direction:column; gap:14px; }
  .frame { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px;
           display:flex; align-items:center; justify-content:center; min-height:300px; }
  .frame img { image-rendering:pixelated; width:100%; max-width:380px; border-radius:4px; }
  .frame pre { font-family:var(--mono); font-size:15px; line-height:1.25; letter-spacing:2px; margin:0; }
  .bar { display:flex; gap:10px; flex-wrap:wrap; }
  .chip { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px 12px; flex:1; min-width:92px; }
  .chip .k { color:var(--muted); font-size:10.5px; text-transform:uppercase; letter-spacing:.08em; }
  .chip .v { font-family:var(--mono); font-size:15px; margin-top:3px; }
  .chip .v.action { color:var(--accent); }
  .chip .v.bad { color:var(--bad); }
  .tally { font-family:var(--mono); font-size:12px; color:var(--muted); margin-left:auto; }
  .tally b { color:var(--ok); }
  .block h3 { margin:0 0 7px; font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }
  .block pre { margin:0 0 16px; background:var(--panel); border:1px solid var(--line); border-radius:8px;
               padding:13px 15px; font-family:var(--mono); font-size:12.5px; white-space:pre-wrap;
               word-break:break-word; max-height:40vh; overflow:auto; }
  .success-banner { background:rgba(75,211,154,.14); color:var(--ok); border:1px solid rgba(75,211,154,.4);
                    border-radius:8px; padding:9px 13px; font-family:var(--mono); font-size:13px; display:none; }
  .success-banner.show { display:block; }
  @media (max-width:860px){ main{grid-template-columns:1fr;} }
</style>
</head>
<body>
<header>
  <h1>Crafter Live</h1>
  <span class="live"><span id="dot" class="dot on"></span><span id="status">connecting</span></span>
  <span class="goal">goal &rarr; <b id="goal">-</b></span>
  <span class="tally" id="tally"></span>
</header>

<main>
  <section class="stage">
    <div id="successBanner" class="success-banner">&#10003; objective reached</div>
    <div class="frame" id="frame"><pre>waiting for the first turn&hellip;</pre></div>
    <div class="bar">
      <div class="chip"><div class="k">model</div><div class="v" id="mModel" style="font-size:12px">-</div></div>
      <div class="chip"><div class="k">trial</div><div class="v" id="mTrial">-</div></div>
      <div class="chip"><div class="k">turn</div><div class="v" id="mTurn">-</div></div>
    </div>
    <div class="bar">
      <div class="chip"><div class="k">action</div><div class="v action" id="mAction">-</div></div>
      <div class="chip"><div class="k">think time</div><div class="v" id="mThink">-</div></div>
      <div class="chip"><div class="k">facing</div><div class="v" id="mFacing">-</div></div>
    </div>
    <div class="block">
      <h3>Inventory / achievements</h3>
      <pre id="mState"></pre>
    </div>
  </section>

  <section class="side">
    <div class="block">
      <h3>Model raw response</h3>
      <pre id="mResponse">-</pre>
    </div>
    <div class="block">
      <h3>Prompt sent this turn</h3>
      <pre id="mPrompt">-</pre>
    </div>
  </section>
</main>

<script>
const POLL_MS = __POLL_MS__;
let lastFrameKey = null;
function esc(s){ return (s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

async function tick(){
  let s;
  try { s = await (await fetch('/state?t=' + Date.now())).json(); }
  catch(e){ setStatus('waiting', false); return; }

  document.getElementById('goal').textContent = s.objective || '-';
  const running = s.status === 'running';
  const done = s.status === 'complete';
  setStatus(done ? 'run complete' : (running ? 'running' : s.status), running, done);

  if (s.model == null){ return; }
  document.getElementById('mModel').textContent = s.model;
  document.getElementById('mTrial').textContent = (s.trial||'-') + ' / ' + (s.num_trials||'-');
  document.getElementById('mTurn').textContent = (s.turn||'-') + ' / ' + (s.max_turns||'-');
  const act = document.getElementById('mAction');
  act.textContent = s.action || '-';
  act.className = 'v action' + (s.parse_ok === false ? ' bad' : '');
  document.getElementById('mThink').textContent =
    (s.think_seconds != null ? s.think_seconds.toFixed(2) + 's' : '-');
  document.getElementById('mFacing').textContent = s.facing || '-';

  // Only touch the frame when the turn actually changes. Rebuilding the <img>
  // on every 700ms poll is what made it flicker/stutter.
  const frameKey = s.model + '|' + s.trial + '|' + s.turn + '|' + (s.frame ? 'img' : 'txt');
  if (frameKey !== lastFrameKey){
    lastFrameKey = frameKey;
    const frame = document.getElementById('frame');
    if (s.frame){
      let img = frame.querySelector('img');
      if (!img){ frame.textContent = ''; img = document.createElement('img'); img.alt = 'state'; frame.appendChild(img); }
      img.src = '/' + s.frame + '?t=' + Date.now();  // fetch the new turn's frame
    } else if (s.map_text){
      frame.innerHTML = '<pre>' + esc(s.map_text) + '</pre>';
    }
  }

  const inv = s.inventory ? Object.entries(s.inventory).map(([k,v])=>k+': '+v).join('\\n') : '';
  const ach = (s.achievements && s.achievements.length) ? s.achievements.join(', ') : 'none yet';
  document.getElementById('mState').textContent =
    'inventory\\n' + (inv || '(empty)') + '\\n\\nachievements\\n' + ach;
  document.getElementById('mResponse').textContent = s.raw_response || '(empty)';
  document.getElementById('mPrompt').textContent = s.prompt || '-';

  if (s.successes != null){
    document.getElementById('tally').innerHTML =
      'solved <b>' + s.successes + '</b> / ' + (s.trials_done||0) + ' trials';
  }
  document.getElementById('successBanner').className =
    'success-banner' + (s.success ? ' show' : '');
}

function setStatus(text, pulsing, done){
  document.getElementById('status').textContent = text;
  const dot = document.getElementById('dot');
  dot.className = 'dot' + (done ? ' done' : (pulsing ? ' on' : ''));
}

tick();
setInterval(tick, POLL_MS);
</script>
</body>
</html>
"""

_PAGE = _PAGE.replace("__POLL_MS__", str(POLL_MS))
