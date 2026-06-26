"""Shared, backend-agnostic interactive web UI for the VLA playgrounds.

One page reused by every instance (pi0.5+LIBERO, MolmoBot+MolmoSpaces, ...): a live
camera stream, a text box to type/replace the instruction, cascading selectors at the
top, play/pause, and a captioned "Save video" export. The page is policy/sim-agnostic —
it talks to a small worker contract, so each instance only implements that contract for
its own VLA + simulator.

A backend "worker" must provide:

    config() -> dict
        {
          "title": str,                              # header + tab title
          "selectors": [                             # ordered, cascading dropdowns
            {"name": "suite", "label": "Suite", "options": ["libero_10", ...]},
            {"name": "task",  "label": "Task",  "depends_on": "suite",
             "options_by": {"libero_10": [{"value": "0", "label": "Task 0 — ...",
                                           "default_prompt": "..."}], ...}},
          ],
          # MolmoBot would instead supply selectors named vla / env / scene.
          "instruction_label": str,                  # optional
          "instruction_placeholder": str,            # optional
        }
        A selector has either static "options" (list of str or {value,label}) or, when it
        "depends_on" another selector, an "options_by" map keyed by the parent's value
        (so the whole catalog ships in one /config call — no extra round-trips). An
        option may carry a "default_prompt" used to prefill the blank instruction.

    snapshot_status() -> dict      # free-form; rendered verbatim as "key: value" rows
    latest_jpeg() -> bytes         # current frame (JPEG)
    set_instruction(text) -> None
    request_reset(selection, instruction="") -> None   # selection = {selector_name: value}
    set_paused(paused: bool) -> None
    save_video(name, speed) -> path|None    # compose/return an mp4 of the run-so-far

`compose_video(frames, prompts, ...)` below is a reusable helper a worker can call from
save_video() to render the captioned, speed-adjustable video; backends that already keep
a per-run rollout.mp4 can just return that path instead.
"""
from __future__ import annotations

import pathlib
import re
import time

import cv2
import imageio
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_file


# ----- reusable video compositor (optional helper for worker.save_video) -----

def compose_video(frames, prompts, name, speed=1.0, runs_dir="runs", out_size=480):
    """Render RGB frames to runs_dir/videos/<name>.mp4 with `prompts[i]` captioned below
    each frame, at fps = 10 * speed. Returns the path, or None if there are no frames."""
    frames = list(frames)
    if not frames:
        return None
    prompts = list(prompts)
    if len(prompts) < len(frames):
        prompts += [prompts[-1] if prompts else ""] * (len(frames) - len(prompts))
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "video"
    vid_dir = pathlib.Path(runs_dir) / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    path = vid_dir / f"{safe}.mp4"
    cap_h, W, font = 64, out_size, cv2.FONT_HERSHEY_SIMPLEX
    fps = max(1, int(round(10 * float(speed))))
    writer = imageio.get_writer(str(path), fps=fps, macro_block_size=None)
    try:
        for img, prompt in zip(frames, prompts):
            fr = cv2.resize(np.asarray(img), (W, out_size), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((out_size + cap_h, W, 3), dtype=np.uint8)
            canvas[:out_size] = fr
            for i, ln in enumerate(_wrap_text(prompt or "—", font, 0.5, 1, W - 16)[:2]):
                cv2.putText(canvas, ln, (8, out_size + 24 + i * 22), font, 0.5,
                            (240, 240, 240), 1, cv2.LINE_AA)
            writer.append_data(canvas)
    finally:
        writer.close()
    return path


def _wrap_text(text, font, scale, thick, max_w):
    out, cur = [], ""
    for w in str(text).split():
        test = (cur + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(test, font, scale, thick)
        if tw > max_w and cur:
            out.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        out.append(cur)
    return out or [""]


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
        body = request.json or {}
        # Accept either {"selection": {...}, "instruction": ...} or a bare selection dict.
        worker.request_reset(body.get("selection", body), body.get("instruction", ""))
        return jsonify(ok=True)

    @app.route("/pause", methods=["POST"])
    def pause():
        worker.set_paused((request.json or {}).get("paused", True))
        return jsonify(ok=True)

    @app.route("/save_video", methods=["POST"])
    def save_video():
        if not hasattr(worker, "save_video"):
            return jsonify(ok=False, error="This backend doesn't support save_video."), 400
        body = request.json or {}
        try:
            speed = float(body.get("speed", 1) or 1)
        except (TypeError, ValueError):
            speed = 1.0
        path = worker.save_video(body.get("name") or "rollout", speed)
        if not path:
            return jsonify(ok=False, error="No frames yet — load a scene and press Play first."), 400
        path = pathlib.Path(path).resolve()
        return send_file(str(path), mimetype="video/mp4", as_attachment=True, download_name=path.name)

    return app


INDEX_HTML = r"""
<!doctype html><html><head><meta charset="utf-8"><title>Interactive VLA</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#111;color:#eee;margin:0;padding:20px}
 .wrap{max-width:1000px;margin:0 auto}
 h1{font-size:18px;font-weight:600}
 .row{display:flex;gap:20px;flex-wrap:wrap}
 #view{width:560px;max-width:100%;background:#000;border:1px solid #333;border-radius:8px;image-rendering:auto}
 .panel{flex:1;min-width:320px}
 input,select,button{font-size:14px;padding:8px;border-radius:6px;border:1px solid #444;background:#1c1c1c;color:#eee}
 input[type=text]{width:100%}
 button{background:#2d6cdf;border:none;cursor:pointer} button.alt{background:#444}
 button:active{transform:scale(0.98)} button:disabled{opacity:0.45;cursor:default}
 input:disabled{opacity:0.5;cursor:not-allowed}
 .green{background:#2ea043} #play.ready{background:#2ea043} button.big{padding:11px;font-weight:600;font-size:15px}
 label{font-size:12px;color:#aaa;display:block;margin:14px 0 5px}
 .hint{color:#888;font-size:12px;margin-top:5px}
 .chip{display:inline-block;background:#1f1f1f;border:1px solid #3a3a3a;border-radius:13px;padding:4px 10px;margin:3px 5px 0 0;font-size:12px;color:#bcd9ff;cursor:pointer}
 .chip:hover{background:#2d6cdf;color:#fff;border-color:#2d6cdf}
 .sel{display:flex;gap:8px;flex-wrap:wrap}.sel>div{flex:1;min-width:130px}
 .toast{min-height:18px;font-size:13px;color:#3fb950;margin:12px 0 6px;transition:opacity .25s;opacity:0}.toast.show{opacity:1}
 .status{background:#000;padding:4px 12px;border-radius:8px;font-size:13px}
 .srow{display:flex;justify-content:space-between;gap:14px;padding:6px 0;border-bottom:1px solid #191919}.srow:last-child{border-bottom:none}
 .srow .k{color:#888}.srow .v{color:#eee;font-family:ui-monospace,monospace;text-align:right;word-break:break-word}
</style></head><body><div class="wrap">
<h1 id="title">Interactive VLA</h1>
<div class="row">
 <img id="view" src="/frame.jpg">
 <div class="panel">
  <div id="selectors"></div>
  <label id="instrlabel">Instruction</label>
  <input type="text" id="instr" placeholder="">
  <div class="hint">Editable while paused; press Play to send it &amp; run.</div>
  <div class="hint" id="canon"></div>
  <div id="examples"></div>
  <div class="row" style="gap:8px;margin-top:16px"><button id="play" class="big" style="flex:2">▶ Play</button><button id="reset" class="alt big" style="flex:1">↻ Reset</button></div>
  <div class="row" style="gap:8px;margin-top:14px;align-items:center"><button id="savevid" class="alt" style="flex:2">💾 Save video</button><label style="margin:0;color:#aaa">speed</label><select id="speed" style="flex:0 0 auto"><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option><option value="8">8×</option></select></div>
  <div class="hint">Send a new instruction any time — corrections or staged subgoals. Save video captions the active prompt under each frame.</div>
  <div class="toast" id="toast"></div>
  <div class="status" id="status"></div>
 </div>
</div></div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>(s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const view=$('view'); let streaming=false;
function pollFrame(){ view.src='/frame.jpg?t='+Date.now(); }
function startPolling(){ view.onload=()=>setTimeout(pollFrame,20); view.onerror=()=>setTimeout(pollFrame,150); pollFrame(); }
(function(){ view.onload=()=>{streaming=true;}; view.onerror=()=>{if(!streaming)startPolling();}; view.src='/stream.mjpg'; setTimeout(()=>{if(!streaming)startPolling();},2500); })();
let toastTimer; function toast(m,c){ const t=$('toast'); t.textContent=m; t.style.color=c||'#3fb950'; t.classList.add('show'); clearTimeout(toastTimer); toastTimer=setTimeout(()=>t.classList.remove('show'),2600); }

let SELS=[];
function selection(){ const s={}; SELS.forEach(x=>{ const el=$('sel_'+x.name); if(el&&el.value!=='') s[x.name]=el.value; }); return s; }
function optList(sel){
 if(sel.options) return sel.options;
 if(sel.depends_on && sel.options_by){ const pv=($('sel_'+sel.depends_on)||{}).value; return sel.options_by[pv]||[]; }
 return [];
}
function fillOptions(sel){
 const el=$('sel_'+sel.name);
 // URL-encode data-dp/data-ex so quotes in prompts/JSON can't break the attribute.
 el.innerHTML=optList(sel).map(o=>{ const v=(typeof o==='string')?o:o.value; const l=(typeof o==='string')?o:(o.label||o.value); const dp=(o&&o.default_prompt)?encodeURIComponent(o.default_prompt):''; const ex=(o&&o.examples)?encodeURIComponent(JSON.stringify(o.examples)):''; return `<option value="${esc(v)}" data-dp="${dp}" data-ex="${ex}">${esc(l)}</option>`; }).join('');
}
function buildSelectors(){
 const wrap=$('selectors'); wrap.innerHTML=''; const lab=document.createElement('label'); lab.textContent=SELS.map(s=>s.label||s.name).join(' · '); wrap.appendChild(lab);
 const bar=document.createElement('div'); bar.className='sel'; wrap.appendChild(bar);
 for(const sel of SELS){ const d=document.createElement('div'); const s=document.createElement('select'); s.id='sel_'+sel.name; s.style.width='100%';
  s.onchange=()=>{ for(const dep of SELS){ if(dep.depends_on===sel.name) fillOptions(dep); } showCanon(); doLoad(); };
  d.appendChild(s); bar.appendChild(d); }
 SELS.filter(s=>!s.depends_on).forEach(fillOptions);
 SELS.filter(s=>s.depends_on).forEach(fillOptions);
 showCanon();
}
function currentOpt(){ const last=SELS[SELS.length-1]; if(!last)return null; const o=($('sel_'+last.name)||{}).selectedOptions; return (o&&o[0])?o[0]:null; }
function showCanon(){ const o=currentOpt();
 const dpRaw=o?o.getAttribute('data-dp'):''; const dp=dpRaw?decodeURIComponent(dpRaw):'';
 $('canon').textContent=dp?('default prompt: '+dp):'';
 let ex=[]; const exRaw=o?o.getAttribute('data-ex'):''; if(exRaw){ try{ ex=JSON.parse(decodeURIComponent(exRaw)); }catch(e){} }
 const box=$('examples');
 if(ex && ex.length>=1){
  box.innerHTML='<div class="hint" style="margin:8px 0 2px">'+ex.length+(ex.length==1?' task':' tasks')+' trained on this scene (click to use):</div>'+ex.map(t=>`<span class="chip">${esc(t)}</span>`).join('');
  box.querySelectorAll('.chip').forEach((c,i)=>c.onclick=()=>{ if(!serverPaused){toast('Pause to change the prompt','#e0a23b');return;} $('instr').value=ex[i]; toast('Prompt set — press Play'); });
 } else box.innerHTML='';
}

async function doLoad(){ toast('Loading scene… (paused — press Play)','#d8a657');
 await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({selection:selection(),instruction:$('instr').value.trim()})}); }
$('reset').onclick=()=>{ toast('Reset — press Play to start','#d8a657'); doLoad(); };
$('instr').addEventListener('keydown',e=>{if(e.key==='Enter')$('play').click();});  // Enter = Play (sends the prompt & runs)
$('savevid').onclick=async()=>{ const def='run_'+new Date().toISOString().slice(0,19).replace(/[:T]/g,'-'); const name=prompt('Name this video:',def); if(name===null)return;
 const speed=$('speed').value,btn=$('savevid'),orig=btn.textContent; btn.disabled=true; btn.textContent='Rendering…'; toast('Rendering video ('+speed+'×)…','#d8a657');
 try{ const r=await fetch('/save_video',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,speed})});
  if(!r.ok){ let j={}; try{j=await r.json();}catch(e){} toast(j.error||'Save failed','#e06c75'); return; }
  const blob=await r.blob(),url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=(name||'run')+'.mp4'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url); toast('Saved ✓ — downloaded + kept on the pod');
 }catch(e){ toast('Save failed: '+e,'#e06c75'); } finally{ btn.disabled=false; btn.textContent=orig; } };
let serverPaused=true, limitReached=false;
function syncPlay(p){ serverPaused=p; const b=$('play'); b.textContent=p?'▶ Play':'⏸ Pause'; b.classList.toggle('ready',p); $('instr').disabled=!p; }  // prompt editable only while paused
$('play').onclick=async()=>{ if(limitReached){toast('Step limit reached — press Reset.','#e0a23b');return;}
 const np=!serverPaused;
 if(!np){ const t=$('instr').value.trim(); if(t) await fetch('/instruction',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})}); }  // pressing Play also sends the typed prompt
 syncPlay(np);
 await fetch('/pause',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({paused:np})}); toast(np?'Paused':'Playing…','#d8a657'); };

function row(k,v){ return `<div class="srow"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`; }
async function poll(){
 try{ const s=await (await fetch('/status')).json();
  if('paused' in s) syncPlay(!!s.paused);
  limitReached=!!s.limit_reached;
  $('status').innerHTML=Object.keys(s).filter(k=>!['paused','limit_reached'].includes(k)).map(k=>row(k,s[k])).join('');
 }catch(e){}
 setTimeout(poll,400);
}
(async()=>{ const c=await (await fetch('/config')).json(); SELS=c.selectors||[];
 document.title=c.title||'Interactive VLA'; $('title').textContent=c.title||'Interactive VLA';
 if(c.instruction_label)$('instrlabel').textContent=c.instruction_label; if(c.instruction_placeholder)$('instr').placeholder=c.instruction_placeholder;
 buildSelectors(); doLoad(); poll(); })();
</script></body></html>
"""
