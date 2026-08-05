import sys
sys.path.insert(0, "/workspace/diffusion_policy")
import numpy as np
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

tf = RotationTransformer('axis_angle', 'rotation_6d')

# the actual axis_angle we observed from the real Lift initial pose
axis_angle_true = np.array([3.11530654, -0.02657987, 0.22830143])
print("angle magnitude (rad):", np.linalg.norm(axis_angle_true), " (pi =", np.pi, ")")

rot6d = tf.forward(axis_angle_true)
recovered = tf.inverse(rot6d)
print("original axis_angle: ", axis_angle_true)
print("recovered axis_angle:", recovered)
print("diff norm:", np.linalg.norm(recovered - axis_angle_true))

# perturb slightly (simulating what a diffusion model's imperfect rotation_6d
# prediction near this pose might look like) and see how much the recovered
# axis_angle jumps
np.random.seed(0)
for i in range(5):
    noisy_rot6d = rot6d + np.random.normal(scale=0.02, size=rot6d.shape)
    recovered_noisy = tf.inverse(noisy_rot6d)
    print(f"trial {i}: small rot6d perturbation -> recovered axis_angle = {recovered_noisy}  (diff from true: {np.linalg.norm(recovered_noisy - axis_angle_true):.4f})")
