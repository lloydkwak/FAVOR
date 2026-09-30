# FAVOR — Where to Intervene? Training-Free Fault Adaptation of Diffusion Policies

**Question.** A diffusion policy was trained on a healthy robot. One joint then fails (locked, or its
range of motion shrinks). Without retraining, *where* in the inference loop should we intervene to
recover task success: during denoising, by selecting among samples, or after sampling?

We treat the faulted robot as a new embodiment and compare interventions at every point of the
loop on LIBERO (Franka Panda, joint-space Diffusion Policy, horizon 16 / 8 executed steps):

| Where | Method | Code |
|---|---|---|
| none | **B1** (no intervention) | `docker/native_joint_policy.py` |
| sample selection | **Random-N**, **Select** (N=32, ε-certificate) | `native_joint_policy.py`, `fault_certificate.py` |
| during denoising | **E-C-I** (fault-constraint projection), **RG-DDPM** (reachability guidance) | `joint_eci_projector.py`, `reach_guided.py`, `native_joint_policy_rg.py` |
| post-hoc retargeting | **B-IK pos / pose** (weighted IK), **Priority IK** (ours: position first, orientation in the null space) | `ik_redistribution.py`, `ik_priority.py`, `native_joint_policy_prio.py` |

The fault (joint and admissible range) is assumed to be given, as in DEFT; robustness to
misspecified fault parameters is evaluated separately (`native_joint_policy_ms.py`).

## Main result (4 tasks × 7 joints per level, n = 20, identical seeds)

| Fault level | B1 | E-C-I | B-IK pos | B-IK pose | Oracle best(pos, pose)† | **Priority IK** |
|---|---|---|---|---|---|---|
| mild | 0.66 | 0.70 | 0.75 | 0.70 | 0.77 | 0.70 |
| moderate | 0.42 | 0.45 | 0.59 | 0.51 | 0.65 | **0.65** |
| severe | 0.28 | 0.31 | 0.45 | 0.39 | 0.54 | **0.58** |
| locked | 0.11 | 0.19 | 0.30 | 0.32 | 0.43 | **0.48** |
| all 112 | 0.37 | – | 0.52 | 0.48 | 0.60 | **0.60** |

† picks the better B-IK setting per condition after seeing the results. Priority IK uses one setting for
every joint and beats both fixed B-IK settings over all 112 conditions (paired McNemar, p < 1e-12).
Full tables and figures: `paper/tables/`, `paper/figs/` (regenerate with `scripts_paper/make_paper_figures.py`).

## Repository layout

~~
docker/                  environment (Dockerfile.libero, compose) + all method and runner code
  fault_injector.py        locked / range_reduced / velocity_limited faults (MuJoCo)
  favor_fault_runner.py    LIBERO rollout runner with fault injection and per-episode logging
  native_joint_policy*.py  policy wrappers: B1, Select, Random-N, E-C-I, B-IK, Priority IK, RG-DDPM, misspecification
  run_libero_fault_sweep_*.py, run_confirm_n50*.py   sweep runners (one JSON per condition)
  sweep_grid_libero*.py    task / joint / severity grids and fixed test seeds
scripts_libero/          data conversion, evaluation, fault-severity design, Layer-1 recoverability analysis
scripts_paper/           figure and table generation from results/
paper/figs, paper/tables generated figures (PDF/PNG) and tables (LaTeX/CSV)
run_*_queue.sh           sweep chains (docker compose, one container per task)
run_libero_all_tasks.sh  policy training for the LIBERO tasks
~~

## Reproduce

~~bash
docker compose -f docker/docker-compose.libero.yml build
./run_libero_all_tasks.sh                                   # train joint-space DPs
# sweeps (inside the container, from /workspace/diffusion_policy)
python /workspace/docker/run_libero_fault_sweep_locked.py <task> {b1,eci,select,random_n,ik,ik_pose}
python /workspace/docker/run_libero_fault_sweep_range.py  <task> <level_idx> {b1,eci,select,random_n,ik,ik_pose}
python /workspace/docker/run_libero_fault_sweep_prio.py   <task> locked | range <level_idx>
python /workspace/docker/run_libero_fault_sweep_x.py      rg <task> locked | range <level_idx>
python /workspace/docker/run_libero_fault_sweep_ms.py     {rs050,rs150,lop010,lom010} <task> locked | range <level_idx>
python /workspace/docker/run_confirm_n50_prio.py          <task>
# figures and tables (host)
python scripts_paper/make_paper_figures.py --results results --out paper
~~

Results (`results/`) are not tracked; every number in `paper/` is regenerated from them.
