import os, math, collections, pathlib, numpy as np
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("PYOPENGL_PLATFORM","egl")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as wcp
DUMMY=[0.0]*6+[-1.0]
def q2aa(q):
    q=list(q)
    if q[3]>1:q[3]=1.0
    if q[3]<-1:q[3]=-1.0
    d=math.sqrt(1-q[3]*q[3]); return np.zeros(3) if math.isclose(d,0) else (np.array(q[:3])*2*math.acos(q[3]))/d
def prep(o):
    img=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(o["agentview_image"][::-1,::-1]),224,224))
    wri=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(o["robot0_eye_in_hand_image"][::-1,::-1]),224,224))
    st=np.concatenate((o["robot0_eef_pos"],q2aa(o["robot0_eef_quat"]),o["robot0_gripper_qpos"]))
    return {"observation/image":img,"observation/wrist_image":wri,"observation/state":st}
ts=benchmark.get_benchmark_dict()["libero_goal"]()
LANG=[ts.get_task(i).language for i in range(10)]
cli=wcp.WebsocketClientPolicy("127.0.0.1",8000)
def run(env,obs,prompt,steps,stop):
    plan=collections.deque(); fired=False
    for _ in range(steps):
        if not plan:
            el=prep(obs); el["prompt"]=prompt; plan.extend(cli.infer(el)["actions"][:5])
        obs,_,d,_=env.step(plan.popleft().tolist())
        if d:
            fired=True
            if stop: break
    return fired,obs
PAIRS=[(1,9),(0,8),(9,8),(7,8),(1,8),(8,1)]  # (A,B); last two share the bowl
T=3; ASTEP=150; BSTEP=200
print("pair (A->B)            | B baseline | B after A | A , B")
for a,b in PAIRS:
    bd=pathlib.Path(get_libero_path("bddl_files"))/ts.get_task(b).problem_folder/ts.get_task(b).bddl_file
    env=OffScreenRenderEnv(bddl_file_name=str(bd),camera_heights=256,camera_widths=256); env.seed(7)
    inits=ts.get_task_init_states(b); base=seq=0
    for t in range(T):
        env.reset(); o=env.set_init_state(inits[t])
        for _ in range(10): o,_,_,_=env.step(DUMMY)
        f,_=run(env,o,LANG[b],BSTEP,True); base+=f
        env.reset(); o=env.set_init_state(inits[t])
        for _ in range(10): o,_,_,_=env.step(DUMMY)
        _,o=run(env,o,LANG[a],ASTEP,False)
        f,_=run(env,o,LANG[b],BSTEP,True); seq+=f
    env.close()
    print("%d->%d  %-2s %2s/%d      %2s/%d     | A=%s ; B=%s"%(a,b,"",base,T,seq,T,LANG[a][:22],LANG[b][:22]))
