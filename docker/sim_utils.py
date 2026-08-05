"""
Shared helper: walk an arbitrary chain of gym.Wrapper layers to find the
underlying MjSim. Replaces the previous hardcoded `self.env.env.sim` in both
FaultInjector and JointActuationWrapper, which silently assumed a fixed
wrapping depth and broke the moment JointActuationWrapper was inserted
between FaultInjector and the raw robomimic env (extra layer -> wrong hop
count -> AttributeError, caught by code review before running).
"""

def find_sim(env):
    node = env
    seen = 0
    while not hasattr(node, "sim"):
        if not hasattr(node, "env"):
            raise AttributeError(f"could not find .sim by walking .env chain (stopped at {type(node)})")
        node = node.env
        seen += 1
        if seen > 10:
            raise RuntimeError("wrapper chain too deep (>10) -- likely a wrapping bug")
    return node.sim
