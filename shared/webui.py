"""Shared interactive web UI for the VLA playgrounds.

One page, reused by every instance (pi0.5+LIBERO, MolmoBot+MolmoSpaces, ...): a live
camera stream, a text box to type/replace the instruction, and **VLA + Env + scene
selectors** at the top. The page is policy/sim-agnostic — it talks to a small backend
contract, so each instance only implements that contract for its own VLA and simulator.

A backend (the per-instance rollout worker) must provide:

    config() -> dict
        {
          "title":  str,                         # shown in the header
          "vlas":   [str, ...],                  # VLA options this server can run
          "envs":   [str, ...],                  # env options this server can run
          "scenes": {env: [str, ...], ...},      # scene labels per env
          "default": {"vla":..., "env":..., "scene":...},
        }
    snapshot_status() -> dict                    # free-form; rendered as key: value lines
    latest_jpeg() -> bytes                       # current frame (JPEG)
    set_instruction(text: str) -> None
    request_reset(selection: dict) -> None       # {"vla":..., "env":..., "scene":...}
    set_paused(paused: bool) -> None

Selectors only advertise what the running server can actually do (a MolmoBot pod won't
list LIBERO). Switching to an option the server can't host is the backend's call to reject.
"""
from __future__ import annotations

import time

from flask import Flask, Response, jsonify, render_template_string, request


def build_app(worker):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(INDEX_HTML)

    @app.route("/config")
    def config():
        return jsonify(worker.config())

    @app.route("/frame.jpg")
    def frame():
        return Response(worker.latest_jpeg(), mimetype="image/jpeg")

    @app.route("/stream.mjpg")
    def stream():
        # MJPEG multipart over one persistent connection: bandwidth-bound, smooth even
        # behind a reverse proxy. The browser renders multipart/x-mixed-replace in an <img>.
        def gen():
            last = None
            while True:
                jpg = worker.latest_jpeg()
                if jpg is not last:
                    last = jpg
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.04)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/status")
    def status():
        return jsonify(worker.snapshot_status())

    @app.route("/instruction", methods=["POST"])
    def instruction():
        worker.set_instruction((request.json or {}).get("text", ""))
        return jsonify(ok=True)

    @app.route("/reset", methods=["POST"])
    def reset():
        worker.request_reset(request.json or {})
        return jsonify(ok=True)

    @app.route("/pause", methods=["POST"])
    def pause():
        worker.set_paused((request.json or {}).get("paused", True))
        return jsonify(ok=True)

    return app


INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Interactive VLA Playground</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#111;color:#eee;margin:0;padding:20px}
 .wrap{max-width:1180px;margin:0 auto}
 h1{font-size:18px;font-weight:600}
 .row{display:flex;gap:20px;flex-wrap:wrap}
 #view{width:1024px;max-width:100%;background:#000;border:1px solid #333;border-radius:8px}
 .panel{flex:1;min-width:300px}
 input,select,button,textarea{font-size:14px;padding:8px;border-radius:6px;border:1px solid #444;background:#1c1c1c;color:#eee}
 input[type=text]{width:100%}
 button{background:#2d6cdf;border:none;cursor:pointer}
 button.alt{background:#444}
 .status{font-family:monospace;font-size:13px;background:#000;padding:10px;border-radius:6px;white-space:pre-wrap;line-height:1.5}
 label{font-size:12px;color:#aaa;display:block;margin:10px 0 4px}
 .hint{color:#888;font-size:12px;margin-top:4px}
 .sel{display:flex;gap:8px;flex-wrap:wrap}
 .sel>div{flex:1;min-width:120px}
</style></head><body><div class="wrap">
<h1 id="title">Interactive VLA Playground</h1>
<img id="view" src="/frame.jpg">
<div class="row" style="margin-top:14px">
 <div class="panel">
  <label>VLA · Environment · Scene</label>
  <div class="sel">
   <div><select id="vla" style="width:100%"></select></div>
   <div><select id="env" style="width:100%"></select></div>
   <div><select id="scene" style="width:100%"></select></div>
  </div>
  <button id="load" style="margin-top:8px;width:100%">Load &amp; start</button>

  <label>Instruction (type anything; objects must exist in the scene)</label>
  <input type="text" id="instr" placeholder="e.g. put the salt shaker in the bowl">
  <div class="row" style="gap:8px;margin-top:8px">
   <button id="send" style="flex:2">Send instruction</button>
   <button id="pause" class="alt" style="flex:1">Pause</button>
   <button id="reset" class="alt" style="flex:1">Reset</button>
  </div>
  <div class="hint">A new instruction forces an immediate replan — use it for corrections or staged subgoals.</div>
 </div>
 <div class="panel">
  <label>Status</label>
  <div class="status" id="status">idle</div>
 </div>
</div></div>
<script>
const $=id=>document.getElementById(id);
const view=$('view');
let streaming=false, CFG=null;
function pollFrame(){ view.src='/frame.jpg?t='+Date.now(); }
function startPolling(){ view.onload=()=>setTimeout(pollFrame,20); view.onerror=()=>setTimeout(pollFrame,150); pollFrame(); }
function startStream(){
 view.onload=()=>{ streaming=true; };
 view.onerror=()=>{ if(!streaming) startPolling(); };
 view.src='/stream.mjpg';
 setTimeout(()=>{ if(!streaming) startPolling(); }, 2500);
}
startStream();
function fill(sel,opts,def){ sel.innerHTML=(opts||[]).map(o=>`<option ${o===def?'selected':''}>${o}</option>`).join(''); }
function refreshScenes(){ const e=$('env').value; const sc=(CFG.scenes||{})[e]||[];
 fill($('scene'),sc,(CFG.default||{}).scene); }
async function loadConfig(){
 const c=await (await fetch('/config')).json(); CFG=c;
 if(c.title){ document.title=c.title; $('title').textContent=c.title; }
 const d=c.default||{};
 fill($('vla'),c.vlas,d.vla); fill($('env'),c.envs,d.env); refreshScenes();
}
$('env').onchange=refreshScenes;
$('load').onclick=async()=>{
 await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({vla:$('vla').value,env:$('env').value,scene:$('scene').value})});
};
$('send').onclick=async()=>{
 await fetch('/instruction',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({text:$('instr').value})});
};
$('instr').addEventListener('keydown',e=>{if(e.key==='Enter')$('send').click();});
let paused=false;
$('pause').onclick=async()=>{ paused=!paused;
 await fetch('/pause',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({paused})});
 $('pause').textContent=paused?'Resume':'Pause';
};
$('reset').onclick=()=>$('load').click();
async function poll(){
 try{ const s=await (await fetch('/status')).json();
  $('status').textContent=Object.entries(s).map(([k,v])=>`${(k+':').padEnd(10)}${v}`).join('\\n');
 }catch(e){}
 setTimeout(poll,400);
}
loadConfig().then(poll);
</script></body></html>
"""
