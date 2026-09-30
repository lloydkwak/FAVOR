"""
(A) Choose B-IK (rot_weight, reg) from KINEMATICS ONLY, by a rule fixed in advance:
    among combos whose p95 of max|dq_free| <= 1.0 rad, minimize mean
    eps = pos_err + 0.05 * rot_err (the certificate/Layer1 metric),
    over 4 tasks x 7 joints x {locked, moderate}, 10 demos, every 4th waypoint.
    Writes the chosen values into docker/ik_redistribution.py.
(B) Layer1 v2 with the chosen IK: 4 tasks x 7 joints x {locked, mild, moderate,
    severe}, all demos, every 4th waypoint. Per condition:
      override_* : EE error if the fault only overrides the joint (what B1 executes)
      ik_*       : EE error left after B-IK redistribution (best a model-based fix reaches)
"""
import sys, re, ast, csv, pathlib, itertools, h5py, numpy as np, torch
sys.path.insert(0, "/workspace/docker")
from fault_kinematics import PandaKinematics
import ik_redistribution as ikr
from sweep_grid_libero_range import TASKS, LEVELS, demo_excursion

BETA = 0.05
kin = PandaKinematics(device="cpu")
kin.set_base_transform(torch.tensor([-0.6, 0.0, 0.0]), torch.eye(3))

def pose_err(qa, qb):
    pa, Ra = kin.forward(qa); pb, Rb = kin.forward(qb)
    return (pa - pb).norm(dim=-1), ikr.rotvec_from_matrix(Ra @ Rb.transpose(-1, -2)).norm(dim=-1)

def load(task, n_demos=None):
    with h5py.File(TASKS[task]["dataset"], "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:n_demos]
        return [f[f"data/{k}/obs/robot0_joint_pos"][:] for k in keys]

def targets(qs, j, keep):   # keep=None -> locked
    tgt, con = [], []
    for q in qs:
        qt = torch.tensor(q[::4], dtype=torch.float32); q0 = float(q[0, j])
        if keep is None:
            c = torch.full((len(qt),), q0)
        else:
            h = keep * float(EXC[TASK_NOW][j]); c = qt[:, j].clamp(q0 - h, q0 + h)
        tgt.append(qt); con.append(c)
    return torch.cat(tgt), torch.cat(con)

TASK_LIST = ["alphabet_soup", "milk", "bowl_ramekin", "bowl_stove"]
EXC = {t: demo_excursion(t) for t in TASK_LIST}

# ---------- (A) parameter selection ----------
combos = [(1.0, 0.0), (0.3, 0.0), (0.1, 0.0), (0.05, 0.0),
          (0.3, 0.05), (0.1, 0.05), (0.05, 0.05), (0.1, 0.1), (0.05, 0.1), (0.1, 0.2)]
cells = []
for TASK_NOW in TASK_LIST:
    qs = load(TASK_NOW, 10)
    for j in range(7):
        for keep in (None, 0.5):
            qt, c = targets(qs, j, keep)
            need = (c - qt[:, j]).abs() > 1e-6
            if need.sum() > 0:
                cells.append((qt[need], c[need], j))
print(f"(A) {len(cells)} cells")
res = []
for rw, rg in combos:
    eps_all, dq_all = [], []
    for qt, c, j in cells:
        q_ik, info = ikr.ik_redistribute(kin, qt, j, c, rot_weight=rw, reg=rg)
        pe, re_ = pose_err(qt, q_ik)
        eps_all.append((pe + BETA * re_).mean().item()); dq_all.append(info["dq_free"])
    dq = torch.cat(dq_all)
    res.append((rw, rg, float(np.mean(eps_all)), dq.mean().item(), torch.quantile(dq, 0.95).item()))
    print(f"  rot_weight={rw:<5} reg={rg:<5} mean eps={res[-1][2]:.4f}  mean max|dq|={res[-1][3]:.3f}  p95={res[-1][4]:.3f}")
ok = [r for r in res if r[4] <= 1.0]
best = min(ok, key=lambda r: r[2]) if ok else min(res, key=lambda r: r[4])
print(f"(A) CHOSEN rot_weight={best[0]} reg={best[1]}  (rule: p95 max|dq|<=1.0 rad, then min mean eps)")
p = pathlib.Path("/workspace/docker/ik_redistribution.py"); s = p.read_text()
s = re.sub(r"^IK_ROT_WEIGHT = .*$", f"IK_ROT_WEIGHT = {best[0]}", s, flags=re.M)
s = re.sub(r"^IK_REG = .*$", f"IK_REG = {best[1]}", s, flags=re.M)
ast.parse(s); p.write_text(s); print("(A) written to docker/ik_redistribution.py")

# ---------- (B) Layer1 v2 ----------
rows = []
levels = [("locked", None)] + [(lv, k) for lv, k in LEVELS]
for TASK_NOW in TASK_LIST:
    qs = load(TASK_NOW)
    for j in range(7):
        for lv, keep in levels:
            qt, c = targets(qs, j, keep)
            need = (c - qt[:, j]).abs() > 1e-6
            q_ov = qt.clone(); q_ov[:, j] = c
            q_ik = qt.clone()
            dq = torch.zeros(len(qt))
            if need.sum() > 0:
                q_new, info = ikr.ik_redistribute(kin, qt[need], j, c[need], rot_weight=best[0], reg=best[1])
                q_ik[need] = q_new; dq[need] = info["dq_free"]
            po, ro = pose_err(qt, q_ov); pi_, ri = pose_err(qt, q_ik)
            rows.append(dict(task=TASK_NOW, joint=f"robot0_joint{j+1}", level=lv,
                             bind_frac=need.float().mean().item(),
                             override_pos_mm=1000 * po.mean().item(), override_rot_deg=float(np.degrees(ro.mean().item())),
                             override_eps=(po + BETA * ro).mean().item(),
                             ik_pos_mm=1000 * pi_.mean().item(), ik_rot_deg=float(np.degrees(ri.mean().item())),
                             ik_eps=(pi_ + BETA * ri).mean().item(),
                             ik_pos_mm_p90=1000 * torch.quantile(pi_, 0.9).item(),
                             ik_dq_mean=dq[need].mean().item() if need.sum() > 0 else 0.0))
            r = rows[-1]
            print(f"  {TASK_NOW:13s} j{j+1} {lv:8s} bind={r['bind_frac']:.2f} override {r['override_pos_mm']:6.1f}mm {r['override_rot_deg']:5.1f}deg"
                  f" -> IK {r['ik_pos_mm']:6.1f}mm {r['ik_rot_deg']:5.2f}deg (p90 {r['ik_pos_mm_p90']:6.1f}mm) dq {r['ik_dq_mean']:.2f}")
out = "/workspace/analysis_out/layer1_v2.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"(B) saved {out}\nLAYER1_V2_DONE")
