import h5py, numpy as np
TASKS = {t: f"/workspace/data/robomimic/datasets/libero_{t}/ph/image_abs.hdf5"
         for t in ("alphabet_soup", "milk", "bowl_ramekin", "bowl_stove")}
LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
HI = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
RNG = HI - LO
print("Panda nominal range [rad]:", RNG.round(3))
for task, path in TASKS.items():
    with h5py.File(path, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        exc, up, dn, T = [], [], [], []
        for k in keys:
            q = f[f"data/{k}/obs/robot0_joint_pos"][:]
            d = q - q[0]
            exc.append(np.abs(d).max(0)); up.append(d.max(0)); dn.append(-d.min(0)); T.append(len(q))
    exc, up, dn = map(np.array, (exc, up, dn))
    print(f"\n=== {task}: {len(keys)} demos, len median {int(np.median(T))} ===")
    print("joint | max|q-q0| med / p90 / max [rad] | up med | down med | med as % nominal")
    for j in range(7):
        print(f"  j{j+1}  | {np.median(exc[:,j]):.3f} / {np.percentile(exc[:,j],90):.3f} / {exc[:,j].max():.3f}"
              f" | {np.median(up[:,j]):.3f} | {np.median(dn[:,j]):.3f} | {100*np.median(exc[:,j])/RNG[j]:.1f}%")
