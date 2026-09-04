"""
Fault-sweep hook for scripts/eval_policy_xpolicylab.py (2단계-5, B1 baseline).

Gated behind the FAULT_SWEEP_HOOK=1 environment variable so the normal
(unmodified) eval.sh path used for the leaderboard reproducibility check
is completely unaffected when this hook is not explicitly enabled.

FOUR STRUCTURAL BUGS FOUND AND FIXED across three pilot runs:

1. run_one_batch_episode only receives `args` (the task_config dict), NOT
   `usr_args` (where fault_arm/fault_joint/fault_type/fault_severity
   actually live). Fix: inject those keys into `args` via a patched
   load_task_args (called synchronously in the parent process, before
   `args` is deepcopied into each worker).

2. main_batch() spawns workers via multiprocessing "spawn", which
   re-imports this script as an ordinary module in each worker WITHOUT
   running `if __name__ == "__main__":`. Fix: apply patches from
   unconditional module-level code (still env-var gated).

3. spawn must PICKLE Process(target=batch_eval_worker). Wrapping
   batch_eval_worker itself in a closure makes it unpicklable. Fix: never
   wrap batch_eval_worker; only wrap load_task_args (parent-process-only,
   never pickled) and run_one_batch_episode (patched by NAME at import
   time in each worker, never itself pickled).

4. (found via a 4th pilot run: "Batch eval completed zero episodes,
   skipped_seeds=150") run_one_batch_episode calls task_env.setup_demo()
   TWICE when expert_check=True: once to pre-validate that the seed's
   SCRIPTED/expert trajectory can complete the task at all (independent
   of any policy or fault), and once for the real policy rollout. The
   previous version of this hook re-attached the injector on EVERY
   setup_demo call, which faulted the expert pre-check too -- but the
   hardcoded expert trajectory was authored assuming full joint range, so
   locking/restricting a joint made the expert check itself fail for
   nearly every seed, exhausting all attempts before any real (policy)
   episode ever ran.
   Fix: use the `expert_check` kwarg (already passed into
   run_one_batch_episode by its caller) to compute which setup_demo call
   index is the REAL one (index 1 of [0,1] if expert_check else index 0
   of [0]), and only attach the fault starting at that call. Earlier
   calls (the expert pre-check) run completely unfaulted, exactly as
   upstream -- seed validity is decided the same way it would be without
   any fault-sweep hook at all.

Fault parameters (read from usr_args, populated by the original script's
own parse_additional_info() from --additional_info=key=value,... --
unchanged):
    fault_arm       "left" | "right"            (required to inject a fault)
    fault_joint     e.g. "fl_joint4"             (required)
    fault_type      "locked" | "range_reduced" | "velocity_limited"
    fault_severity  float in (0,1], only used for range_reduced/velocity_limited

If fault_arm/fault_joint/fault_type are absent, episodes run exactly as
upstream (== B1, no fault active) even with the hook enabled.
"""


def _fault_params_from_args(args: dict) -> dict | None:
    arm = args.get("fault_arm")
    joint = args.get("fault_joint")
    ftype = args.get("fault_type")
    if not arm or not joint or not ftype:
        return None
    severity = args.get("fault_severity")
    severity = float(severity) if severity is not None else None
    return {"arm_tag": arm, "joint_name": joint, "fault_type": ftype, "severity": severity}


_FAULT_KEYS = ("fault_arm", "fault_joint", "fault_type", "fault_severity")


def patch_load_task_args(module_globals: dict) -> None:
    original_load_task_args = module_globals["load_task_args"]

    def patched_load_task_args(usr_args, *a, **kw):
        args, embodiment_name = original_load_task_args(usr_args, *a, **kw)
        for key in _FAULT_KEYS:
            if key in usr_args:
                args[key] = usr_args[key]
        return args, embodiment_name

    module_globals["load_task_args"] = patched_load_task_args


def patch_run_one_batch_episode(module_globals: dict) -> None:
    from envs.fault_injection.fault_injector_sapien import FaultInjector

    original_run_one_batch_episode = module_globals["run_one_batch_episode"]

    def patched_run_one_batch_episode(worker_id, local_episode_id, seed_value, task_name,
                                       task_env, args, model_client, **kwargs):
        fault_params = _fault_params_from_args(args)

        if fault_params is None:
            return original_run_one_batch_episode(
                worker_id, local_episode_id, seed_value, task_name, task_env, args, model_client, **kwargs
            )

        # Determine which setup_demo() call is the REAL (policy-rollout)
        # one -- see bug (4) in the module docstring. If expert_check is
        # enabled there are 2 calls (index 0 = expert pre-check, index 1
        # = real); otherwise there is only 1 call (index 0 = real).
        expert_check_enabled = bool(kwargs.get("expert_check", True))
        real_call_index = 1 if expert_check_enabled else 0

        injector_holder = {}
        call_counter = {"n": 0}
        original_setup_demo = task_env.setup_demo

        def wrapped_setup_demo(*a, **kw):
            this_call_index = call_counter["n"]
            call_counter["n"] += 1
            result = original_setup_demo(*a, **kw)
            if this_call_index == real_call_index:
                if "injector" in injector_holder:
                    injector_holder["injector"].detach()
                injector = FaultInjector(task_env.robot, **fault_params)
                injector.attach()
                injector_holder["injector"] = injector
            return result

        import os as _os
        debug = _os.environ.get("FAULT_SWEEP_DEBUG") == "1"
        if debug:
            print(f"[FAULT_SWEEP_DEBUG] episode={local_episode_id} "
                  f"fault_params={fault_params} real_call_index={real_call_index} "
                  f"(expert_check={expert_check_enabled})")

        task_env.setup_demo = wrapped_setup_demo
        try:
            result = original_run_one_batch_episode(
                worker_id, local_episode_id, seed_value, task_name, task_env, args, model_client, **kwargs
            )
        finally:
            task_env.setup_demo = original_setup_demo
            if "injector" in injector_holder:
                inj = injector_holder["injector"]
                if debug:
                    print(f"[FAULT_SWEEP_DEBUG] episode={local_episode_id} "
                          f"fault_info={inj.get_fault_info()} "
                          f"current_qpos_this_arm={inj.get_current_qpos()}")
                inj.detach()

        return result

    module_globals["run_one_batch_episode"] = patched_run_one_batch_episode


def apply_fault_sweep_hooks(module_globals: dict) -> None:
    patch_load_task_args(module_globals)
    patch_run_one_batch_episode(module_globals)
    print("[FAULT_SWEEP_HOOK] load_task_args + run_one_batch_episode patched "
          "(spawn-safe; fault applied only to the real policy rollout, "
          "expert pre-check always runs unfaulted) -- params read from usr_args")
