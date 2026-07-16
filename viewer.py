"""
viewer.py
=========

Builds a single self-contained ``viewer.html`` inside the run folder so you can
replay exactly what each model did: the game state, the prompt it saw, its raw
response, the action it chose, the turn number, and how long it thought.

    python viewer.py                 # uses config.yaml to find the run
    python viewer.py my_config.yaml
    python viewer.py --results runs/gather_wood_10x10/results.json

Open the generated ``runs/<name>/viewer.html`` in a browser. It references the
PNG frames alongside it (and falls back to the ASCII map if frames were off).
Navigate with the on-screen controls or the Left/Right arrow keys; press Space
to play/pause.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# =============================================================================
#  Data extraction (slim the transcript to what the page needs)
# =============================================================================
def build_view_data(results: dict) -> dict:
    models = []
    for name, rec in results["models"].items():
        trials = []
        for t in rec.get("trials", []):
            trials.append({
                "trial": t["trial"],
                "success": t["success"],
                "success_turn": t["success_turn"],
                "turns_used": t["turns_used"],
                "turns": [{
                    "turn": tr["turn"],
                    "frame": tr.get("frame"),
                    "map_text": tr.get("map_text", ""),
                    "prompt": tr.get("prompt", ""),
                    "raw_response": tr.get("raw_response", ""),
                    "action": tr.get("parsed_action", ""),
                    "parse_ok": tr.get("parse_ok", True),
                    "think": tr.get("think_seconds", 0.0),
                    "pos": tr.get("player_pos", []),
                    "facing": tr.get("facing", ""),
                    "inventory": tr.get("inventory", {}),
                    "unlocked": tr.get("achievements_unlocked", []),
                    "success": tr.get("success", False),
                } for tr in t["turns"]],
            })
        models.append({
            "name": name,
            "slug": rec.get("slug", name),
            "error": rec.get("error"),
            "trials": trials,
        })
    return {
        "experiment": results.get("experiment", "experiment"),
        "objective": results.get("objective", {}).get("label", ""),
        "world_size": results.get("world_size", []),
        "models": models,
    }


def render_html(view: dict) -> str:
    payload = json.dumps(view).replace("</", "<\\/")
    return _TEMPLATE.replace("/*__DATA__*/", payload)


# =============================================================================
#  The page  (restrained "lab replay" aesthetic; monospace for game data)
# =============================================================================
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Crafter replay</title>
<style>
  :root{
    --bg:#eceef1; --panel:#ffffff; --ink:#1f2733; --muted:#6a7684;
    --line:#d5dae1; --accent:#33578a; --amber:#b4632a;
    --ok:#2f7d55; --fail:#b5433f; --chip:#eef2f7;
    --radius:10px;
  }
  @media (prefers-color-scheme: dark){
    :root{ --bg:#14171c; --panel:#1c2129; --ink:#e6eaf0; --muted:#93a0af;
      --line:#2b323d; --accent:#7aa2d6; --amber:#d98a54; --ok:#5cbd8a;
      --fail:#e06b68; --chip:#232a33; }
  }
  *{box-sizing:border-box}
  body{
    margin:0; background:var(--bg); color:var(--ink);
    font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    font-size:14px; line-height:1.5;
  }
  .mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
  header{
    display:flex; flex-wrap:wrap; align-items:center; gap:14px;
    padding:14px 20px; border-bottom:1px solid var(--line); background:var(--panel);
  }
  header h1{
    font-size:15px; font-weight:700; letter-spacing:-.01em; margin:0;
    display:flex; align-items:baseline; gap:10px;
  }
  header h1 .goal{font-weight:500; color:var(--muted)}
  header h1 .goal b{color:var(--accent); font-weight:700}
  .selects{display:flex; gap:10px; margin-left:auto; align-items:center}
  select{
    font:inherit; padding:6px 10px; border:1px solid var(--line);
    border-radius:8px; background:var(--panel); color:var(--ink); max-width:280px;
  }
  .badge{
    padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700;
    border:1px solid transparent;
  }
  .badge.ok{color:var(--ok); background:color-mix(in srgb,var(--ok) 14%,transparent); border-color:color-mix(in srgb,var(--ok) 40%,transparent)}
  .badge.fail{color:var(--fail); background:color-mix(in srgb,var(--fail) 14%,transparent); border-color:color-mix(in srgb,var(--fail) 40%,transparent)}

  main{
    display:grid; grid-template-columns:minmax(300px,420px) 1fr; gap:18px;
    padding:18px 20px; align-items:start;
  }
  @media (max-width:820px){ main{grid-template-columns:1fr} }
  .card{
    background:var(--panel); border:1px solid var(--line);
    border-radius:var(--radius); overflow:hidden;
  }
  .card h2{
    margin:0; padding:10px 14px; font-size:11px; letter-spacing:.09em;
    text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--line);
    display:flex; justify-content:space-between; align-items:center;
  }
  .stage{padding:14px; display:flex; justify-content:center; background:
    repeating-conic-gradient(color-mix(in srgb,var(--line) 30%,transparent) 0% 25%, transparent 0% 50%) 50%/22px 22px;}
  .stage img{
    image-rendering:pixelated; width:100%; max-width:360px; height:auto;
    border-radius:6px; display:block;
  }
  pre.map{margin:0; padding:14px; font-size:15px; line-height:1.15; letter-spacing:2px; overflow:auto}

  .meta{display:flex; flex-wrap:wrap; gap:8px; padding:12px 14px; border-top:1px solid var(--line)}
  .kv{background:var(--chip); border-radius:8px; padding:6px 10px; font-size:12px}
  .kv b{color:var(--muted); font-weight:600; margin-right:6px; text-transform:uppercase; letter-spacing:.05em; font-size:10px}
  .action-chip{color:var(--amber); font-weight:700}
  .action-chip.badaction{color:var(--fail)}
  .unlocked{color:var(--ok); font-weight:700}

  .col{display:flex; flex-direction:column; gap:18px}
  .body{padding:14px; white-space:pre-wrap; word-break:break-word; max-height:340px; overflow:auto}
  details.card > summary{list-style:none; cursor:pointer}
  details.card > summary::-webkit-details-marker{display:none}
  details.card > summary h2{cursor:pointer}
  details.card > summary h2::after{content:"\25B8"; color:var(--muted)}
  details.card[open] > summary h2::after{content:"\25BE"}

  /* Signature element: the turn filmstrip */
  .strip-wrap{padding:14px 20px 24px; border-top:1px solid var(--line); background:var(--panel)}
  .controls{display:flex; align-items:center; gap:10px; margin-bottom:12px}
  .controls button{
    font:inherit; font-weight:600; padding:7px 14px; border:1px solid var(--line);
    border-radius:8px; background:var(--panel); color:var(--ink); cursor:pointer;
  }
  .controls button:hover{border-color:var(--accent)}
  .controls .turnlabel{font-variant-numeric:tabular-nums; color:var(--muted); margin-left:6px}
  .strip{display:flex; gap:3px; overflow-x:auto; padding-bottom:6px}
  .tick{
    flex:0 0 auto; width:14px; height:26px; border-radius:3px; cursor:pointer;
    background:color-mix(in srgb,var(--muted) 28%,transparent); border:1px solid transparent;
  }
  .tick.do{background:color-mix(in srgb,var(--accent) 55%,transparent)}
  .tick.bad{background:color-mix(in srgb,var(--amber) 65%,transparent)}
  .tick.win{background:var(--ok)}
  .tick.cur{outline:2px solid var(--ink); outline-offset:1px}
  .legend{display:flex; gap:16px; margin-top:10px; font-size:11px; color:var(--muted); flex-wrap:wrap}
  .legend span{display:inline-flex; align-items:center; gap:6px}
  .swatch{width:12px; height:12px; border-radius:3px; display:inline-block}
  button:focus-visible, .tick:focus-visible, select:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
  .empty{padding:40px; text-align:center; color:var(--muted)}
</style>
</head>
<body>
<header>
  <h1><span id="expname"></span><span class="goal">goal:&nbsp;<b id="goal"></b></span></h1>
  <div class="selects">
    <select id="modelSel" aria-label="Model"></select>
    <select id="trialSel" aria-label="Trial"></select>
    <span id="outcome" class="badge"></span>
  </div>
</header>

<main>
  <div class="card">
    <h2>Game state <span id="frameNo" class="mono"></span></h2>
    <div id="stage" class="stage"></div>
    <div id="meta" class="meta"></div>
  </div>

  <div class="col">
    <details class="card" open>
      <summary><h2>Prompt &mdash; what the model saw</h2></summary>
      <div id="prompt" class="body mono"></div>
    </details>
    <div class="card">
      <h2>Model response &mdash; raw</h2>
      <div id="response" class="body mono"></div>
    </div>
  </div>
</main>

<div class="strip-wrap">
  <div class="controls">
    <button id="prev">&#9664; Prev</button>
    <button id="play">&#9654; Play</button>
    <button id="next">Next &#9654;</button>
    <span class="turnlabel" id="turnlabel"></span>
  </div>
  <div id="strip" class="strip"></div>
  <div class="legend">
    <span><i class="swatch" style="background:color-mix(in srgb,var(--muted) 28%,transparent)"></i>move / other</span>
    <span><i class="swatch" style="background:color-mix(in srgb,var(--accent) 55%,transparent)"></i>interact (do)</span>
    <span><i class="swatch" style="background:color-mix(in srgb,var(--amber) 65%,transparent)"></i>unparsed reply</span>
    <span><i class="swatch" style="background:var(--ok)"></i>goal reached</span>
  </div>
</div>

<script>
const DATA = /*__DATA__*/;
const $ = id => document.getElementById(id);
let mi = 0, ti = 0, k = 0, playing = false, timer = null;

const models = () => DATA.models;
const curModel = () => models()[mi];
const curTrial = () => curModel().trials[ti];
const curTurns = () => curTrial() ? curTrial().turns : [];

function initHeader(){
  $("expname").textContent = DATA.experiment;
  $("goal").textContent = DATA.objective || "\u2014";
  const ms = $("modelSel");
  ms.innerHTML = models().map((m,i)=>`<option value="${i}">${m.name}${m.error?" (load error)":""}</option>`).join("");
  ms.onchange = e => { mi = +e.target.value; ti = 0; k = 0; buildTrials(); render(); };
}

function buildTrials(){
  const ts = $("trialSel");
  const trials = curModel().trials;
  ts.innerHTML = trials.map((t,i)=>`<option value="${i}">Trial ${t.trial} \u2014 ${t.success?"solved":"failed"}</option>`).join("")
    || `<option>no trials</option>`;
  ts.onchange = e => { ti = +e.target.value; k = 0; render(); };
}

function tickClass(tr, isWin){
  if(isWin) return "win";
  if(!tr.parse_ok) return "bad";
  if(tr.action === "do") return "do";
  return "";
}

function buildStrip(){
  const strip = $("strip");
  const trial = curTrial();
  if(!trial){ strip.innerHTML = ""; return; }
  strip.innerHTML = curTurns().map((tr,i)=>{
    const isWin = trial.success_turn === tr.turn;
    return `<div class="tick ${tickClass(tr,isWin)}" data-i="${i}" title="turn ${tr.turn+1}: ${tr.action}" tabindex="0"></div>`;
  }).join("");
  strip.querySelectorAll(".tick").forEach(el=>{
    el.onclick = ()=>{ k = +el.dataset.i; render(); };
    el.onkeydown = e=>{ if(e.key==="Enter"){ k=+el.dataset.i; render(); } };
  });
}

function render(){
  const trial = curTrial();
  const badge = $("outcome");
  if(!trial){
    $("stage").innerHTML = `<div class="empty">No trials recorded for this model.</div>`;
    $("meta").innerHTML=""; $("prompt").textContent=""; $("response").textContent="";
    badge.className="badge"; badge.textContent=""; $("strip").innerHTML=""; $("turnlabel").textContent="";
    return;
  }
  badge.className = "badge " + (trial.success?"ok":"fail");
  badge.textContent = trial.success ? `solved @ turn ${trial.success_turn+1}` : `failed (${trial.turns_used} turns)`;

  const turns = curTurns();
  k = Math.max(0, Math.min(k, turns.length-1));
  const tr = turns[k];

  const stage = $("stage");
  if(tr.frame){
    stage.innerHTML = `<img alt="game state at turn ${tr.turn+1}" src="${tr.frame}">`;
  } else {
    stage.innerHTML = `<pre class="map mono">${escapeHtml(tr.map_text)}</pre>`;
  }
  $("frameNo").textContent = `turn ${tr.turn+1} / ${turns.length}`;

  const inv = Object.entries(tr.inventory).map(([a,b])=>`${a} ${b}`).join(", ") || "empty";
  const badAction = !tr.parse_ok ? " badaction" : "";
  $("meta").innerHTML = `
    <span class="kv"><b>action</b><span class="action-chip${badAction}">${tr.action}${tr.parse_ok?"":" (unparsed \u2192 fallback)"}</span></span>
    <span class="kv"><b>think</b>${tr.think.toFixed(2)}s</span>
    <span class="kv"><b>pos</b>(${tr.pos.join(", ")})</span>
    <span class="kv"><b>facing</b>${tr.facing}</span>
    <span class="kv"><b>inventory</b>${inv}</span>
    ${tr.unlocked.length?`<span class="kv"><b>unlocked</b><span class="unlocked">${tr.unlocked.join(", ")}</span></span>`:""}
  `;

  $("prompt").textContent = tr.prompt;
  $("response").textContent = tr.raw_response || "(empty response)";
  $("turnlabel").textContent = `turn ${tr.turn+1} of ${turns.length}`;

  buildStrip();
  document.querySelectorAll(".tick").forEach((el,i)=> el.classList.toggle("cur", i===k));
}

function step(delta){
  const n = curTurns().length;
  if(!n) return;
  k = (k + delta + n) % n;
  render();
}
function togglePlay(){
  playing = !playing;
  $("play").innerHTML = playing ? "&#10074;&#10074; Pause" : "&#9654; Play";
  if(playing){ timer = setInterval(()=>{
    if(k >= curTurns().length-1){ clearInterval(timer); playing=false; $("play").innerHTML="&#9654; Play"; return; }
    step(1);
  }, 550); } else { clearInterval(timer); }
}
function escapeHtml(s){ return (s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

$("prev").onclick = ()=>step(-1);
$("next").onclick = ()=>step(1);
$("play").onclick = togglePlay;
document.addEventListener("keydown", e=>{
  if(e.target.tagName==="SELECT") return;
  if(e.key==="ArrowLeft"){ step(-1); e.preventDefault(); }
  else if(e.key==="ArrowRight"){ step(1); e.preventDefault(); }
  else if(e.code==="Space"){ togglePlay(); e.preventDefault(); }
});

initHeader();
buildTrials();
render();
</script>
</body>
</html>
"""


# =============================================================================
#  Entry point
# =============================================================================
def resolve_paths(args) -> tuple[Path, Path]:
    if args.results:
        results_path = Path(args.results)
        return results_path, results_path.parent / "viewer.html"
    from config import load_config
    cfg = load_config(args.config)
    return cfg.results_path, cfg.run_dir / "viewer.html"


def build_viewer(results_path: Path, out_path: Path) -> Path:
    results = json.loads(Path(results_path).read_text())
    view = build_view_data(results)
    out_path.write_text(render_html(view))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an HTML replay viewer.")
    ap.add_argument("config", nargs="?", default="config.yaml")
    ap.add_argument("--results", help="path to results.json (overrides config)")
    args = ap.parse_args()

    results_path, out_path = resolve_paths(args)
    out = build_viewer(results_path, out_path)
    print(f"Viewer written to {out}")
    print("Open it in a browser (keep it inside the run folder so frames load).")


if __name__ == "__main__":
    main()
