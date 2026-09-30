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
docker/                     environment (Dockerfile.libero, compose) + all method and runner code
  assets/franka_panda.urdf    Panda kinematics used by every IK-based method (FK matched to the sim to ~1 mm)
  fault_injector.py           locked / range_reduced / velocity_limited faults (MuJoCo)
  favor_fault_runner.py       LIBERO rollout runner with fault injection and per-episode logging
  native_joint_policy*.py     policy wrappers: B1, Select, Random-N, E-C-I, B-IK, Priority IK, RG-DDPM, misspecification
  run_libero_fault_sweep_*.py sweep runners (one JSON per condition, per-episode success)
  run_*confirm_n50*.py        n=50 replications
  sweep_grid_libero*.py       task / joint / severity grids and fixed test seeds
  tests/                      unit tests (E-C-I projection, Priority IK)
third_party/                pinned upstream commits + our patch to diffusion_policy (joint-space configs)
scripts_libero/             LIBERO -> robomimic conversion, evaluation, severity design, Layer-1 analysis
scripts_paper/              figures and tables from results/
paper/figs, paper/tables    generated figures (PDF/PNG) and tables (LaTeX/CSV)
run_main_sweeps.sh          every main-result sweep
run_x_queue.sh, run_ms_queue.sh, run_n50_queue.sh   RG-DDPM / budget variant, misspecification, n=50
run_libero_all_tasks.sh     policy training
~~

## Setup

~~bash
git clone https://github.com/lloydkwak/FAVOR.git && cd FAVOR
bash third_party/setup_third_party.sh        # LIBERO @8f1084e, diffusion_policy @20537a5 + favor.patch
docker compose -f docker/docker-compose.libero.yml build
~~

Data: download the LIBERO demonstrations with LIBERO's own download script, then convert each task to the
robomimic layout the joint-space pipeline consumes (absolute joint-position actions):

~~bash
python scripts_libero/convert_libero_to_robomimic.py --libero-file <LIBERO demo .hdf5> \
    --out data/robomimic/datasets/libero_<task>/ph/image_abs.hdf5
~~

| Task (this repo) | LIBERO task definition (suite / bddl) | demos |
|---|---|---|
| `libero_alphabet_soup` | `libero_object/pick_the_alphabet_soup_and_place_it_in_the_basket.bddl` | 50 |
| `libero_milk` | `libero_object/pick_the_milk_and_place_it_in_the_basket.bddl` | 50 |
| `libero_bowl_ramekin` | `libero_spatial/pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl` | 50 |
| `libero_bowl_stove` | `libero_spatial/pick_the_akita_black_bowl_on_the_stove_and_place_it_on_the_plate.bddl` | 50 |

The matching LIBERO demo file is `<suite>/<task>_demo.hdf5` (same name as the bddl file). `drawer` was also converted and trained but is excluded from all reported results.

## Reproduce

~~bash
./run_libero_all_tasks.sh                     # train joint-space Diffusion Policies (4 tasks)
./run_main_sweeps.sh                          # all main sweeps -> results/
./run_x_queue.sh rg locked range:0            # RG-DDPM
./run_x_queue.sh prio_b015 range:2            # Priority IK + motion budget (appendix)
./run_ms_queue.sh none rs050:range:0 rs150:range:0 lop010:locked lom010:locked   # misspecified faults
./run_n50_queue.sh                            # fresh-seed n=50 replication
# Layer-1 kinematic analysis (writes analysis_out/layer1_v2.csv; part (A) already fixed the B-IK weights)
docker compose -f docker/docker-compose.libero.yml run --rm -v $PWD/analysis_out:/workspace/analysis_out libero \
    python /workspace/scripts_libero/ik_select_and_layer1v2.py
# figures and tables (host; needs numpy + matplotlib)
python scripts_paper/make_paper_figures.py --results results --out paper
# unit tests (container)
docker compose -f docker/docker-compose.libero.yml run --rm libero python /workspace/docker/tests/test_ik_priority.py
~~

Results (`results/`) are not tracked; every number in `paper/` is regenerated from them.
