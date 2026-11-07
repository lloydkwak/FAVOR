"""Builds the pouring scene with MjSpec: table, xArm7 (+ gripper) on the table, cup rigidly
attached to the gripper, bowl on the table (optional scale plate), spherical beads, side + wrist cameras.

All dimensions come from the config tree (config/default.yaml). The compiled model and a few
name/id maps are returned in a SceneHandles object.
"""
import math
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .config import resolve

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTS = [f"act{i}" for i in range(1, 8)]

# Per-gripper layout: body the cup and the wrist camera attach to, the TCP site of the gripper
# and the grip point of the cup handle along that body's z (approach) axis. Fingers close along y.
GRIPPERS = {
    "xarm_gripper": dict(body="xarm_gripper_base_link", tcp_site="link_tcp", grip_z=0.150, actuator="gripper",
                         # Measured on the Menagerie model (finger pads, gripper base frame): actuator command ->
                         # pad centre |y| and pad centre z (mean of the two pads). Pads are 4 mm thick.
                         pad_table=dict(ctrl=[0, 50, 100, 150, 200, 255],
                                        y=[0.0464, 0.0399, 0.0318, 0.0230, 0.0139, 0.0037],
                                        z=[0.1422, 0.1467, 0.1509, 0.1536, 0.1549, 0.1545]),
                         pad_half_thickness=0.002),
    "robotiq_2f85": dict(body="rq_base", tcp_site="rq_pinch", grip_z=0.135, actuator="rq_fingers_actuator"),
}


def grip_command(cfg):
    """(gripper actuator command, pad-centre z in the gripper frame) for mount 'grip': the pads touch
    the cup's outer wall. Explicit robot.gripper.closed_ctrl (a number) overrides the command."""
    g = cfg["robot"]["gripper"]
    info = GRIPPERS[g["type"]]
    r_out = cfg["cup"]["inner_radius"] + cfg["cup"]["wall"]
    tab = info.get("pad_table")
    if tab is None:
        if not isinstance(g.get("closed_ctrl"), (int, float)):
            raise ValueError(f"set robot.gripper.closed_ctrl and cup.grip_z for {g['type']}")
        return float(g["closed_ctrl"]), float(cfg["cup"].get("grip_z", info["grip_z"]))
    y_pad = r_out + info["pad_half_thickness"]
    if y_pad > tab["y"][0]:
        raise ValueError(f"cup outer diameter {2 * r_out * 1000:.0f} mm exceeds the gripper opening")
    ctrl = float(np.interp(-y_pad, [-v for v in tab["y"]], tab["ctrl"]))
    if isinstance(g.get("closed_ctrl"), (int, float)):
        ctrl = float(g["closed_ctrl"])
    z = float(np.interp(ctrl, tab["ctrl"], tab["z"]))
    return ctrl, z

# contype / conaffinity bits: 1 = default (robot, table, cup, bowl), 2 = particles.
PARTICLE_CONTYPE, PARTICLE_CONAFF = 2, 3


@dataclass
class SceneHandles:
    model: mujoco.MjModel
    spec: mujoco.MjSpec
    cfg: dict
    arm_qpos: np.ndarray = None       # qpos addresses of joint1..7
    arm_dof: np.ndarray = None        # dof addresses
    arm_act: np.ndarray = None        # actuator ids
    gripper_act: int = -1
    gripper_ctrl: float = 0.0       # finger command that holds the cup
    particle_qpos: np.ndarray = None  # (N,) start qpos address of each particle free joint
    particle_dof: np.ndarray = None
    particle_body: np.ndarray = None
    particle_geom: np.ndarray = None
    particle_mass: float = 0.0
    sites: dict = field(default_factory=dict)
    bodies: dict = field(default_factory=dict)
    geom_groups: dict = field(default_factory=dict)  # name -> np.array of geom ids
    cameras: dict = field(default_factory=dict)


def _quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return q


def lookat_quat(pos, target, up):
    """MuJoCo camera orientation (x right, y up, looking along -z) for a look-at spec."""
    pos, target, up = (np.asarray(v, dtype=np.float64) for v in (pos, target, up))
    f = target - pos
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    y = np.cross(right, f)
    R = np.stack([right, y, -f], axis=1)
    return _quat_from_mat(R)


def camera_pose_from_cfg(ext):
    """Returns (pos, quat) in the parent frame, MuJoCo camera convention."""
    if ext["type"] == "lookat":
        return np.asarray(ext["pos"], float), lookat_quat(ext["pos"], ext["target"], ext.get("up", [0, 0, 1]))
    if ext["type"] == "matrix":
        T = np.asarray(ext["T"], float).reshape(4, 4)
        R = T[:3, :3].copy()
        if ext.get("convention", "opencv") == "opencv":     # x right, y down, z forward -> MuJoCo
            R[:, 1] *= -1
            R[:, 2] *= -1
        return T[:3, 3].copy(), _quat_from_mat(R)
    raise ValueError(ext["type"])


def _ring(body, name, r_in, wall, z0, height, n, rgba, **kw):
    """Hollow cylinder wall of n boxes, inner radius r_in, from z0 to z0+height (body frame)."""
    half_w = (r_in + wall) * math.tan(math.pi / n) * 1.02
    for k in range(n):
        a = 2 * math.pi * k / n
        g = body.add_geom()
        g.name = f"{name}_wall{k:02d}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [wall / 2, half_w, height / 2]
        g.pos = [(r_in + wall / 2) * math.cos(a), (r_in + wall / 2) * math.sin(a), z0 + height / 2]
        g.quat = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
        g.rgba = rgba
        for k2, v in kw.items():
            setattr(g, k2, v)


VIS = dict(contype=0, conaffinity=0, group=1)


def _collision_shell(body, name, r_in, wall, bottom, z_inner_bottom, height, n, extra):
    """Invisible collision shell (group 3, zero density) for a container: same inner surface as the
    visual geometry, walls and bottom thickened outward / downward by `extra` so small fast beads
    cannot tunnel through a 3-4 mm wall under the soft CG contact solution."""
    hide = dict(group=3, density=0.0)
    w, b = wall + extra, bottom + extra
    body.add_geom(name=f"{name}_bottom_col", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r_in + w, b / 2, 0],
                  pos=[0, 0, z_inner_bottom - b / 2], rgba=[0.5, 0.5, 0.5, 0.0], **hide)
    _ring(body, f"{name}_col", r_in, w, z_inner_bottom - b, height + b, n, [0.5, 0.5, 0.5, 0.0], **hide)


def _add_table(spec, cfg):
    t = cfg["table"]
    wb = spec.worldbody
    tb = wb.add_body(name="table", pos=[t["center_offset"][0], t["center_offset"][1], 0.0])
    g = tb.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[t["size"][0] / 2, t["size"][1] / 2, t["thickness"] / 2],
                    pos=[0, 0, -t["thickness"] / 2])
    spec.add_material(name="table_mat", rgba=t["rgba"])
    g.material = "table_mat"
    leg_h = t["height"] - t["thickness"]
    for sx in (-1, 1):
        for sy in (-1, 1):
            tb.add_geom(name=f"table_leg_{sx}{sy}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, leg_h / 2],
                        pos=[sx * (t["size"][0] / 2 - 0.05), sy * (t["size"][1] / 2 - 0.05), -t["thickness"] - leg_h / 2],
                        rgba=[0.3, 0.3, 0.3, 1], contype=0, conaffinity=0)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], pos=[0, 0, -t["height"]],
                material="floor_mat")


def _add_bowl(spec, cfg):
    """Bowl on the table top, or on a scale plate when bowl.scale_height > 0. The body 'bowl_base'
    (static) is what reset() moves to the sampled bowl position."""
    b = cfg["bowl"]
    sh = float(b.get("scale_height", 0.0))
    sc = spec.worldbody.add_body(name="bowl_base", pos=[cfg["task"]["grid_center"][0], cfg["task"]["grid_center"][1], 0])
    if sh > 0:
        sc.add_geom(name="scale_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[b["scale_size"][0], b["scale_size"][1], sh / 2],
                    pos=[0, 0, sh / 2], rgba=[0.15, 0.15, 0.15, 1])
    bowl = sc.add_body(name="bowl", pos=[0, 0, sh])
    spec.add_material(name="bowl_mat", rgba=b["rgba"])
    bowl.add_geom(name="bowl_bottom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[b["inner_radius"] + b["wall"], b["bottom"] / 2, 0], pos=[0, 0, b["bottom"] / 2], material="bowl_mat",
                  **VIS)
    _ring(bowl, "bowl", b["inner_radius"], b["wall"], 0.0, b["depth"] + b["bottom"], b["n_wall_segments"], b["rgba"], **VIS)
    # The collision bottom extends into the table / scale plate (all static: no contact between them).
    _collision_shell(bowl, "bowl", b["inner_radius"], b["wall"], b["bottom"], b["bottom"], b["depth"],
                     b["n_wall_segments"], cfg["physics"].get("collision_shell_extra", 0.008))
    bowl.add_site(name="bowl_inner_bottom", pos=[0, 0, b["bottom"]], size=[0.003, 0, 0], rgba=[1, 0, 0, 1], group=4)


def _add_cup(spec, cfg, gripper_body, grip_z):
    """Cup frame: origin = inner bottom centre, z = cup axis (up when upright), x = towards the
    pouring lip, y completes the frame.

    mount = grip (default): the fingers pinch the cup body from the side, no handle. Gripper
        approach axis (gz) horizontal when the cup is upright, cup axis = -gx, cup centre on the
        approach axis at the pad centre (grip_z, from grip_command), pads at cup.grip_height above
        the inner bottom. cup.lip_yaw_deg places the lip around the cup axis: 0 = far side from the
        palm (+gz), 90 = towards the +gy finger.
    mount = hang: gripper points down, the fingers pinch a vertical handle plate on the cup side.
    mount = side: gripper approach horizontal, handle block on the cup side facing the flange."""
    c = cfg["cup"]
    gb = spec.body(gripper_body)
    hl = c.get("handle", {"length": 0.05, "width": 0.02, "height": 0.03})
    r_out = c["inner_radius"] + c["wall"]
    total_h = c["inner_depth"] + c["bottom"]
    mount = c.get("mount", "grip")
    handle_reach = hl["length"] - 0.015
    handle_size = None
    if mount == "grip":
        a = math.radians(c.get("lip_yaw_deg", 0.0))
        xc = np.array([0.0, math.sin(a), math.cos(a)])        # lip direction in the gripper frame
        zc = np.array([-1.0, 0.0, 0.0])                       # cup axis
        R = np.stack([xc, np.cross(zc, xc), zc], axis=1)
        pos = [c.get("grip_height", 0.025), 0.0, grip_z]
    elif mount == "hang":
        R = np.diag([1.0, -1.0, -1.0])
        mouth_gz = grip_z - c.get("mouth_above_grip", 0.015)
        pos = [handle_reach + r_out, 0.0, mouth_gz + c["inner_depth"]]   # inner-bottom centre in gripper frame
        handle_pos = [-(r_out + hl["length"] / 2), 0, c["inner_depth"] - hl["height"] / 2]
        handle_size = [hl["length"] / 2, hl["width"] / 2, hl["height"] / 2]
    elif mount == "side":
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
        pos = [(c["inner_depth"] - c["bottom"]) / 2, 0.0, grip_z + handle_reach + r_out]
        handle_pos = [-(r_out + hl["length"] / 2), 0, (c["inner_depth"] - c["bottom"]) / 2]
        handle_size = [hl["length"] / 2, hl["width"] / 2, hl["height"] / 2]
    else:
        raise ValueError(mount)
    cup = gb.add_body(name="cup", pos=pos, quat=_quat_from_mat(R))
    cup.gravcomp = 1.0 if cfg["physics"].get("gravity_comp", True) else 0.0
    spec.add_material(name="cup_mat", rgba=c["rgba"])
    n = c["n_wall_segments"]
    half_w = (c["inner_radius"] + c["wall"]) * math.tan(math.pi / n) * 1.02
    vols = {"bottom": math.pi * r_out ** 2 * c["bottom"], "wall": c["wall"] * 2 * half_w * total_h,
            "handle": 0.0 if handle_size is None else 8 * handle_size[0] * handle_size[1] * handle_size[2]}
    rho = c["mass"] / (vols["bottom"] + n * vols["wall"] + vols["handle"])
    cup.add_geom(name="cup_bottom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r_out, c["bottom"] / 2, 0],
                 pos=[0, 0, -c["bottom"] / 2], material="cup_mat", mass=rho * vols["bottom"], **VIS)
    _ring(cup, "cup", c["inner_radius"], c["wall"], -c["bottom"], total_h, n, c["rgba"], mass=rho * vols["wall"], **VIS)
    if handle_size is not None:
        cup.add_geom(name="cup_handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=handle_size, pos=handle_pos,
                     material="cup_mat", mass=rho * vols["handle"])
    _collision_shell(cup, "cup", c["inner_radius"], c["wall"], c["bottom"], 0.0, c["inner_depth"], n,
                     cfg["physics"].get("collision_shell_extra", 0.008))
    cup.add_site(name="cup_mouth", pos=[0, 0, c["inner_depth"]], size=[0.003, 0, 0], rgba=[0, 1, 0, 1], group=4)
    cup.add_site(name="cup_inner_bottom", pos=[0, 0, 0], size=[0.003, 0, 0], rgba=[0, 0, 1, 1], group=4)
    cup.add_site(name="cup_lip", pos=[c["inner_radius"] + c["wall"], 0, c["inner_depth"]], size=[0.003, 0, 0],
                 rgba=[1, 0, 1, 1], group=4)
    return cup


def _add_particles(spec, cfg):
    p = cfg["particles"]
    n = int(p["count"])
    r = p["diameter_mm"] / 2000.0
    m = p["total_mass_g"] / 1000.0 / max(n, 1)
    spec.add_material(name="particle_mat", rgba=p["rgba"])
    for i in range(n):
        b = spec.worldbody.add_body(name=f"p{i:04d}", pos=[0, 0, 1.0 + 0.01 * i])
        b.add_freejoint(name=f"p{i:04d}_free")
        g = b.add_geom(name=f"p{i:04d}_g", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0], mass=m,
                       material="particle_mat")
        g.contype, g.conaffinity = PARTICLE_CONTYPE, PARTICLE_CONAFF
        g.condim = int(p["condim"])
        g.friction = p["friction"]
        g.solref = p["solref"]
        g.solimp = list(p["solimp"]) + [0.5, 2.0]
    return n, m


def _add_cameras(spec, cfg, gripper_body):
    for name in ("agentview", "eye_in_hand"):
        c = cfg["cameras"][name]
        parent = spec.worldbody if c["parent"] == "world" else spec.body(gripper_body)
        pos, quat = camera_pose_from_cfg(c["extrinsics"])
        cam = parent.add_camera(name=name, pos=pos.tolist(), quat=quat.tolist())
        intr = c["intrinsics"]
        # fx/fy/cx/cy intrinsics are realised at render time (PourEnv.render: square-pixel render
        # with fovy from fy, then crop/resize to put the principal point at (cx, cy)); MuJoCo's own
        # sensorsize/principal path did not reproduce the OpenCV projection in tests/test_camera.py.
        if "fx" in intr:
            cam.fovy = math.degrees(2 * math.atan(0.5 * intr["height"] / float(intr["fy"])))
        else:
            cam.fovy = float(intr["fovy_deg"])
        cam.resolution = [intr["width"], intr["height"]]
        if c.get("body", {}).get("enabled", False):
            _add_camera_body(spec, parent, name, pos, quat, c["body"])


def _add_camera_body(spec, parent, name, pos, quat, b):
    """Physical camera housing (and optional bracket) for a camera, so it shows up in the viewer and
    in the other camera's images and is part of the collision checks.

    b.size     : housing [width (camera x), height (camera y), depth (along the optical axis)], m.
                 RealSense D435: 90 x 25 x 25 mm.
    b.lens     : position of the colour lens on the front face relative to the face centre, in the
                 camera frame [x, y] (x = image right, y = image up). The camera pose (extrinsics)
                 is the colour lens; the housing is placed so that lens sits at this offset.
    b.lenses_x : x of the visible lens discs on the front face (decoration only, no collision).
    b.bracket  : {from: point on the parent body (parent frame), at_x: x on the housing bottom face
                 where the bracket meets it (camera frame), width, thickness}.
    The housing lies behind the image plane (the camera looks along -z), so it never occludes its
    own image."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    R = R.reshape(3, 3)
    w, hgt, dep = b.get("size", [0.090, 0.025, 0.025])
    lx, ly = b.get("lens", [0.0, 0.0])
    spec.add_material(name=f"{name}_body_mat", rgba=b.get("rgba", [0.70, 0.71, 0.73, 1.0]))
    spec.add_material(name=f"{name}_bracket_mat", rgba=[0.08, 0.08, 0.09, 1.0])
    face = np.asarray(pos, float) - R @ np.array([lx, ly, 0.0])          # front-face centre
    centre = face + R @ np.array([0.0, 0.0, dep / 2 + 0.001])
    parent.add_geom(name=f"{name}_housing", type=mujoco.mjtGeom.mjGEOM_BOX, size=[w / 2, hgt / 2, dep / 2],
                    pos=centre.tolist(), quat=list(quat), material=f"{name}_body_mat", mass=b.get("mass", 0.072))
    for k, x in enumerate(b.get("lenses_x", [lx])):
        lp = face + R @ np.array([x, ly, 0.0005])
        parent.add_geom(name=f"{name}_lens{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0045, 0.0004, 0],
                        pos=lp.tolist(), quat=list(quat), rgba=[0.02, 0.02, 0.03, 1], contype=0, conaffinity=0,
                        group=1, mass=0.0)
    br = b.get("bracket")
    if br:
        a = np.asarray(br["from"], float)
        top = centre + R @ np.array([br.get("at_x", 0.0), -hgt / 2, 0.0])
        v = top - a
        L = float(np.linalg.norm(v))
        zax = v / L
        ref = R[:, 2] if abs(np.dot(R[:, 2], zax)) < 0.9 else R[:, 0]   # bracket plate faces the optical axis
        xax = np.cross(ref, zax)
        xax /= np.linalg.norm(xax)
        Rb = np.stack([xax, np.cross(zax, xax), zax], axis=1)
        parent.add_geom(name=f"{name}_bracket", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[br.get("width", 0.020) / 2, br.get("thickness", 0.004) / 2, L / 2],
                        pos=(a + v / 2).tolist(), quat=_quat_from_mat(Rb).tolist(),
                        material=f"{name}_bracket_mat", mass=b.get("bracket_mass", 0.02))


def build_scene(cfg, n_particles=None):
    cfg = dict(cfg)
    if n_particles is not None:
        cfg["particles"] = dict(cfg["particles"], count=n_particles)
    mdir = os.path.join(resolve(cfg["assets"]["menagerie_dir"]))
    gtype = cfg["robot"]["gripper"]["type"]
    ginfo = GRIPPERS[gtype]
    if gtype == "xarm_gripper":
        spec = mujoco.MjSpec.from_file(os.path.join(mdir, "ufactory_xarm7", "xarm7.xml"))
    else:
        spec = mujoco.MjSpec.from_file(os.path.join(mdir, "ufactory_xarm7", "xarm7_nohand.xml"))
        rq = mujoco.MjSpec.from_file(os.path.join(mdir, "robotiq_2f85", "2f85.xml"))
        site = spec.site("attachment_site")
        spec.attach(rq, site=site, prefix="rq_")    # TODO(real): add the adapter flange thickness
    spec.modelname = "favor_xarm7_pour"

    ph = cfg["physics"]
    spec.option.timestep = ph["timestep"]
    spec.option.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{ph['integrator'].upper()}")
    spec.option.solver = getattr(mujoco.mjtSolver, f"mjSOL_{ph.get('solver', 'newton').upper()}")
    spec.option.iterations = ph["iterations"]
    spec.option.tolerance = ph.get("tolerance", 1e-8)
    spec.option.ls_iterations = ph["ls_iterations"]
    spec.option.noslip_iterations = ph["noslip_iterations"]
    spec.option.cone = getattr(mujoco.mjtCone, f"mjCONE_{ph['cone'].upper()}")
    spec.option.impratio = ph["impratio"]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    q = cfg["cameras"].get("quality", {})
    spec.visual.quality.offsamples = int(q.get("offsamples", 0))
    spec.visual.quality.shadowsize = int(q.get("shadow_size", 2048))
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.25, 0.27, 0.30], rgb2=[0.20, 0.22, 0.25], width=256, height=256)
    mat = spec.add_material(name="floor_mat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
    mat.texrepeat = [8, 8]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.85, 0.87, 0.90], rgb2=[0.55, 0.57, 0.60], width=256, height=256)

    # Robot mount, limits, servo gains, gravity compensation.
    base = spec.body("link_base")
    rb = cfg["robot"]
    base.pos = rb["base_pos"]
    yaw = math.radians(rb["base_yaw_deg"])
    base.quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    ctl = cfg["control"]
    for i, jn in enumerate(ARM_JOINTS):
        j = spec.joint(jn)
        lo, hi = (math.radians(v) for v in rb["joint_limits_deg"][i])
        j.range = [lo, hi]
        j.damping = [float(ctl["joint_damping"][i]), 0.0, 0.0]
        j.armature = ctl["armature"]
        j.frictionloss = ctl["frictionloss"]
        a = spec.actuator(ARM_ACTS[i])
        a.ctrlrange = [lo, hi]
        kp, kv = ctl["kp"][i], ctl["kv"][i]
        a.gainprm = [kp] + [0] * 9
        a.biasprm = [0, -kp, -kv] + [0] * 7
        a.forcerange = [-ctl["force_limit"][i], ctl["force_limit"][i]]
    if ph.get("gravity_comp", True):
        stack = [base]
        while stack:
            b = stack.pop()
            b.gravcomp = 1.0
            stack.extend(b.bodies)

    for nm, lp, ld, dif, sh in (("key_light", [0.3, -0.6, 1.6], [0, 0.3, -1], 0.6, True),
                                ("fill_light", [0.8, 0.8, 1.2], [-0.4, -0.4, -1], 0.3, False)):
        L = spec.worldbody.add_light(name=nm, pos=lp, dir=ld, diffuse=[dif] * 3, castshadow=sh)
        L.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    _add_table(spec, cfg)
    from .randomization import add_table_textures
    add_table_textures(spec)                 # unused unless randomization.enabled
    _add_bowl(spec, cfg)
    if cfg["cup"].get("mount", "grip") == "grip":
        grip_ctrl, grip_z = grip_command(cfg)
    else:
        grip_ctrl, grip_z = float(cfg["robot"]["gripper"].get("closed_ctrl", 200.0)), cfg["cup"].get("grip_z", ginfo["grip_z"])
    _add_cup(spec, cfg, ginfo["body"], grip_z)
    n, m_p = _add_particles(spec, cfg)
    _add_cameras(spec, cfg, ginfo["body"])
    gb = spec.body(ginfo["body"])
    gb.add_site(name="gripper_frame", pos=[0, 0, 0], size=[0.002, 0, 0], group=4)

    # The cup is part of the gripper: no contacts between the cup and the gripper bodies.
    grip_bodies = []
    stack = [gb]
    while stack:
        b = stack.pop()
        grip_bodies.append(b.name)
        stack.extend(b.bodies)
    for bn in grip_bodies:
        if bn != "cup":
            spec.add_exclude(bodyname1="cup", bodyname2=bn)
    for bn in ("link7",):
        spec.add_exclude(bodyname1="cup", bodyname2=bn)

    # Keyframe from the Menagerie file has the wrong nq once particles exist.
    for k in list(spec.keys):
        spec.delete(k)

    model = spec.compile()
    h = SceneHandles(model=model, spec=spec, cfg=cfg)
    h.gripper_ctrl = grip_ctrl
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in ARM_JOINTS]
    h.arm_qpos = np.array([model.jnt_qposadr[j] for j in jid])
    h.arm_dof = np.array([model.jnt_dofadr[j] for j in jid])
    h.arm_act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in ARM_ACTS])
    h.gripper_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ginfo["actuator"])
    pj = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"p{i:04d}_free") for i in range(n)]
    h.particle_qpos = np.array([model.jnt_qposadr[j] for j in pj], dtype=int)
    h.particle_dof = np.array([model.jnt_dofadr[j] for j in pj], dtype=int)
    h.particle_body = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"p{i:04d}") for i in range(n)], dtype=int)
    h.particle_geom = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"p{i:04d}_g") for i in range(n)], dtype=int)
    h.particle_mass = m_p
    for s in ("cup_mouth", "cup_inner_bottom", "cup_lip", "bowl_inner_bottom", "gripper_frame", ginfo["tcp_site"]):
        h.sites[s] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
    h.sites["gripper_tcp"] = h.sites[ginfo["tcp_site"]]
    for b in ("cup", "bowl", "bowl_base", "table", "link_base", ginfo["body"]):
        h.bodies[b] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
    h.bodies["gripper"] = h.bodies[ginfo["body"]]
    for c in ("agentview", "eye_in_hand"):
        h.cameras[c] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c)

    # Geom groups for safety checks and metrics.
    def subtree_geoms(body_id):
        ids = []
        for g in range(model.ngeom):
            b = model.geom_bodyid[g]
            while b > 0 and b != body_id:
                b = model.body_parentid[b]
            if b == body_id:
                ids.append(g)
        return np.array(ids, dtype=int)

    robot_all = subtree_geoms(h.bodies["link_base"])
    cup_g = subtree_geoms(h.bodies["cup"])
    h.geom_groups["robot"] = np.array([g for g in robot_all if g not in set(cup_g)], dtype=int)
    h.geom_groups["cup"] = cup_g
    h.geom_groups["bowl"] = subtree_geoms(h.bodies["bowl"])
    sg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "scale_top")
    h.geom_groups["scale"] = np.array([sg] if sg >= 0 else [], dtype=int)
    h.geom_groups["table"] = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top"),
                                       mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")])
    h.geom_groups["particles"] = h.particle_geom
    return h
