import os

import trimesh
import warp as wp
import math
import numpy as np
from pathlib import Path
from .common import *

try:
    from .tcdm_utils import (
        get_tcdm_trajectory,
        TCDM_MESH_PATHS,
        TCDM_TRAJ_PATHS,
        TCDM_OBJ_NAMES,
    )
    from tcdm.envs import asset_abspath
except ImportError as e:
    print(f"WARNING: did not import TCDM package due to error: {e}")
    asset_abspath = lambda x: x
    get_tcdm_trajectory = TCDM_TRAJ_PATHS = TCDM_OBJ_NAMES = []
    TCDM_MESH_PATHS = {}


OBJ_PATHS = {
    ObjectType.CYLINDER_MESH: "cylinder.stl",
    ObjectType.CUBE_MESH: "cube.stl",
    ObjectType.OCTPRISM_MESH: "octprism-2.stl",
    ObjectType.ELLIPSOID_MESH: "ellipsoid.stl",
}


def quat_multiply(a, b):
    return np.array(
        (
            a[3] * b[0] + b[3] * a[0] + a[1] * b[2] - b[1] * a[2],
            a[3] * b[1] + b[3] * a[1] + a[2] * b[0] - b[2] * a[0],
            a[3] * b[2] + b[3] * a[2] + a[0] * b[1] - b[0] * a[1],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        )
    )


def add_joint(
    builder,
    pos,
    ori,
    joint_type,
    joint_axis=(0.0, 0.0, 1.0),
    body_name="object",
    limit_lower=-2 * np.pi * 3.0,
    limit_upper=2 * np.pi * 3.0,
    parent=-1,
    stiffness=0.0,
    damping=0.75,
):
    """Add a joint to the builder"""

    body_link = builder.add_body(
        # parent=parent,
        origin=wp.transform(pos, wp.quat_identity()),
        name=body_name,
        armature=0.0,
    )
    if joint_type == wp.sim.JOINT_FREE:
        builder.add_joint_free(body_link, wp.transform(pos, ori), parent=parent, name="obj_joint")
    elif joint_type == wp.sim.JOINT_REVOLUTE:
        builder.add_joint_revolute(
            parent,
            body_link,
            wp.transform(pos, ori),
            wp.transform_identity(),
            axis=joint_axis,
            target_ke=stiffness,  # 1.0
            target_kd=damping,  # 1.5
            limit_ke=100.0,
            limit_kd=150.0,
            limit_lower=limit_lower,
            limit_upper=limit_upper,
            name="obj_joint",
        )
    else:
        raise ValueError("Invalid joint")

    return body_link


def add_object(
    builder,
    link,
    object_type,
    rot=(1.0, 0.0, 0.0, 0.0),
    density=1.0,
    contact_ke=1e4,
    contact_kd=1e2,
    xform=None,
    scale=2,
    model_path=None,
):
    if object_type in OBJ_PATHS:
        add_mesh(
            builder,
            link,
            OBJ_PATHS[object_type],
            contact_ke=contact_ke,
            contact_kd=contact_kd,
            density=density,
        )

    elif object_type in TCDM_TRAJ_PATHS:
        add_tcdm_mesh(
            builder,
            link,
            model_path,
            rot,
            density,
            contact_ke,
            contact_kd,
            1.3,
        )
    elif object_type is ObjectType.RECTANGULAR_PRISM_FLAT:
        add_box(
            builder,
            link,
            size=(0.1, 0.3, 0.1),
            density=density,
            ke=contact_ke,
            kd=contact_kd,
            xform=xform,
        )
    elif object_type is ObjectType.RECTANGULAR_PRISM:
        add_box(
            builder,
            link,
            size=(0.15, 0.15, 0.2),
            density=density,
            ke=contact_ke,
            kd=contact_kd,
            xform=xform,
        )
    elif object_type is ObjectType.CUBE:
        add_box(
            builder,
            link,
            size=(0.15, 0.15, 0.15),
            density=density,
            ke=contact_ke,
            kd=contact_kd,
            xform=xform,
        )
    elif object_type is ObjectType.SPHERE:
        add_sphere(builder, link, 0.15, contact_ke, contact_kd)
    elif object_type is ObjectType.OCTPRISM:
        start_shape_count = len(builder.shape_geo_type)
        hy = 0.06  # dividing circumference into 8 segments
        hx = (math.sqrt(2) + 1) * hy  # length of box to extend to other side of prism
        hz = 0.15
        size = (hx * scale, hy * scale, hz * scale)
        add_box(
            builder,
            link,
            size=size,
            density=density,
            ke=contact_ke,
            kd=contact_kd,
            xform=xform,
        )
        rot = wp.quat_from_axis_angle((0, 0, 1), math.pi / 4)
        add_box(
            builder,
            link,
            rot=rot,
            size=size,
            density=density,
            ke=contact_ke,
            kd=contact_kd,
            xform=xform,
        )
        rot = wp.quat_from_axis_angle((0, 0, 1), math.pi / 2)
        add_box(
            builder,
            link,
            rot=rot,
            size=size,
            density=density,
            ke=contact_ke,
            kd=contact_kd,
            xform=xform,
        )
        rot = wp.quat_from_axis_angle((0, 0, 1), 3 * math.pi / 4)
        add_box(
            builder,
            link,
            rot=rot,
            size=size,
            density=density,
            ke=contact_ke,
            kd=contact_kd,
            xform=xform,
        )
        end_shape_count = len(builder.shape_geo_type)
        for i in range(start_shape_count, end_shape_count):
            for j in range(i + 1, end_shape_count):
                builder.shape_collision_filter_pairs.add((i, j))


def add_mesh(builder, link, obj_path, ke=2.0e5, kd=1e4, density=100.0):
    obj_path = str(Path(os.path.dirname(__file__)).parent / "envs" / "assets" / obj_path)
    mesh, scale = parse_mesh(obj_path)
    geom_size = (0.5, 0.5, 0.5)

    builder.add_shape_mesh(
        body=link,
        pos=(0.0, 0.0, 0.0),  # in Z-up frame, transform applied already
        # rot=(0.70710678, 0.0, 0.0, 0.70710678),
        mesh=mesh,
        scale=geom_size,
        density=density,
        ke=ke,
        kd=kd,
    )


def add_tcdm_mesh(
    builder,
    link,
    obj_path,
    rot=(1.0, 0.0, 0.0, 0.0),
    density=100.0,
    ke=2.0e5,
    kd=1e4,
    scale=1,
):
    #'hammer-use1' --> ['hammer', 'use1']
    obj_stl_path = asset_abspath(obj_path)
    mesh, _ = parse_mesh(obj_stl_path)
    geom_size = (scale, scale, scale)
    builder.add_shape_mesh(
        body=link,
        pos=(0.0, 0.0, 0.0),  # in Z-up frame, transform applied already
        rot=rot,
        mesh=mesh,
        scale=geom_size,
        density=density,
        ke=ke,
        kd=kd,
    )


def add_sphere(builder, link, radius=0.15, ke=1e5, kd=1e3):
    builder.add_shape_sphere(
        body=link,
        pos=(0.0, 0.0, 0.0),  # in Z-up frame, transform applied already
        radius=radius,
        density=100.0,
        ke=ke,
        kd=kd,
    )


def add_box(
    builder,
    link,
    rot=(0.0, 0.0, 0.0, 1.0),
    size=(0.25, 0.25, 0.25),
    density=100.0,
    ke=1e4,
    kd=100,
    xform=None,
):
    hx, hy, hz = size

    if xform is not None:
        q = wp.transform_get_rotation(xform)
        t = wp.transform_get_rotation(xform)
        rot = quat_multiply(q, rot)

    builder.add_shape_box(
        body=link,
        pos=(0.0, 0.0, hz),  # in Z-up frame, transform applied already
        rot=rot,
        hx=hx,
        hy=hy,
        hz=hz,
        density=density,
        ke=ke,
        kd=kd,
    )


def create_shadow_hand(
    builder,
    action_type,
    floating_base=False,
    stiffness=1000.0,
    damping=0.0,
    hand_start_position=(0.01, 0.30, 0.125),
    hand_start_orientation=(0, 0, 0),
    base_joint=None,
    collapse_joints=True,
):
    if action_type is ActionType.TORQUE:
        stiffness, damping = 0.0, 0.0
    if floating_base:
        if len(hand_start_orientation) == 3:
            hand_start_orientation = wp.quat_rpy(*hand_start_orientation)
        elif len(hand_start_orientation) == 4:
            hand_start_orientation = wp.quat(*hand_start_orientation)
        xform = wp.transform(hand_start_position, hand_start_orientation)
    else:
        xform = wp.transform(
            np.array((0.1, 0.15, 0.0)),
            # wp.quat_rpy(-np.pi / 2 * 3, np.pi * 0.75, np.pi / 2),  # thumb down
            # wp.quat_rpy(-np.pi / 2 * 3, np.pi * 0.75, np.pi / 2 * 3),  # thumb up (default) palm orthogonal to gravity
            wp.quat_rpy(-np.pi / 2 * 3, np.pi * 1.25, np.pi / 2 * 3),  # thumb up palm down
        )
    wp.sim.parse_urdf(
        os.path.join(
            os.path.dirname(__file__),
            "envs/assets/shadowhand/shadowhand.urdf",
        ),
        builder,
        xform=xform,
        floating=floating_base,
        base_joint=base_joint,
        density=1e3,
        armature=0.01,
        stiffness=stiffness,
        damping=damping,
        shape_ke=1.0e3,
        shape_kd=1.0e2,
        shape_kf=1.0e2,
        shape_mu=0.5,
        limit_ke=1.0e4,
        limit_kd=1.0e1,
        enable_self_collisions=False,
    )
    if collapse_joints:
        builder.collapse_fixed_joints()

    # ensure all joint positions are within limits
    q_offset = 7 if floating_base else 0
    for i in range(2, 22):
        x, y = 0.5, 0.5
        builder.joint_q[i + q_offset] = x * (builder.joint_limit_lower[i]) + y * (builder.joint_limit_upper[i])
        builder.joint_target[i] = builder.joint_q[i + q_offset]

        if action_type is ActionType.TORQUE:
            builder.joint_target_ke[i] = 0.0
            builder.joint_target_kd[i] = 0.0
        else:
            builder.joint_target_ke[i] = 5000.0
            builder.joint_target_kd[i] = 10.0


########### Allegro Hand poses (base orientation)
# thumb up palm down
# wp.quat_rpy(-np.pi / 2 * 3, np.pi * 1.25, np.pi / 2 * 3)
# thumb up (default) palm orthogonal to gravity
# wp.quat_rpy(-np.pi / 2 * 3, np.pi * 0.75, np.pi / 2 * 3),
# thumb up, palm facing left
# wp.quat_rpy(-np.pi / 2 * 3, np.pi * 0.75, np.pi / 2 * 3),
# thumb up, palm facing right
# wp.quat_rpy(np.pi * 0.0, np.pi * 1.0, np.pi * -0.25),
# thumb down
# wp.quat_rpy(-np.pi / 2 * 3, np.pi * 0.75, np.pi / 2),

DMANIP_ENVS_PATH = os.path.join(os.path.split(os.path.dirname(__file__))[0], "envs")


def create_allegro_hand(
    builder,
    action_type,
    floating_base=False,
    stiffness=1000.0,
    damping=0.0,
    hand_start_position=(0.0, 0.30, 0.0),
    hand_start_orientation=(-np.pi / 2, np.pi * 0.75, np.pi / 2),
    # (np.pi * 0.0, np.pi * 1.0, np.pi * -0.25),
    base_joint=None,
    collapse_joints=True,
):
    if action_type is ActionType.TORQUE:
        stiffness, damping = 0.0, 0.0
    xform = wp.transform(hand_start_position, wp.quat_rpy(*hand_start_orientation))
    wp.sim.parse_urdf(
        os.path.join(
            DMANIP_ENVS_PATH,
            "assets/isaacgymenvs/kuka_allegro_description/allegro.urdf",
        ),
        builder,
        xform=xform,
        floating=floating_base,
        base_joint=base_joint,
        density=1e3,
        armature=0.1,  # 0.01
        stiffness=stiffness,
        damping=damping,
        shape_ke=1.0e3,
        shape_kd=1.0e2,
        shape_kf=1.0e2,
        shape_mu=0.5,
        shape_thickness=0.001,
        limit_ke=1.0e4,
        limit_kd=1.0e1,
        enable_self_collisions=False,
    )
    if collapse_joints:
        builder.collapse_fixed_joints()

    # ensure all joint positions are within limits
    q_offset = 7 if floating_base else 0
    fixed_base_offset = 3 if base_joint is not None else 0

    for i in range(fixed_base_offset, 16 + fixed_base_offset):
        x, y = 0.65, 0.35
        builder.joint_q[i + q_offset] = x * (builder.joint_limit_lower[i]) + y * (builder.joint_limit_upper[i])
        builder.joint_target[i] = builder.joint_q[i + q_offset]

        builder.joint_target_ke[i] = stiffness
        builder.joint_target_kd[i] = damping


class ObjectModel:
    object_id: int = None

    def __init__(
        self,
        object_type,
        base_pos=(0.0, 0.075, 0.0),
        base_ori=(np.pi / 2, 0.0, 0.0),
        joint_type=wp.sim.JOINT_FREE,
        contact_ke=1e3,
        contact_kd=1e2,
        scale=0.4,
        density=1e2,
        stiffness=0.0,
        damping=0.5,
        base_body_name="base",
        model_path=None,
        floating=None,
        base_joint=None,
    ):
        self.object_type = object_type
        self.object_name = object_type.name.lower()
        self.base_pos = base_pos
        if len(base_ori) == 3:
            self.base_ori = tuple(x for x in wp.quat_rpy(*base_ori))
        elif len(base_ori) == 4:
            self.base_ori = base_ori
        self.joint_type = joint_type
        self.contact_ke = contact_ke
        self.contact_kd = contact_kd
        self.scale = scale
        self.stiffness = stiffness
        self.damping = damping
        self.model_path = model_path  # TCDM_MESH_PATHS.get(self.object_type)
        self.base_joint_name = base_body_name + "_joint"
        self.density = density
        self.base_joint = base_joint
        self.floating = floating
        # if self.model_path is not None:
        #     self.tcdm_trajectory, self.dex_trajectory = get_tcdm_trajectory(self.object_type)
        # else:
        self.tcdm_trajectory = self.dex_trajectory = None

    def remesh(self, builder):
        remeshed_path = os.path.join(
            os.path.split(os.path.split(__file__)[0])[0], "envs", "assets", "remeshed", self.object_type.name
        )
        if self.object_id is not None:
            remeshed_path += "_" + str(self.object_id)
        remeshed_path += ".npz"
        if os.path.exists(remeshed_path):
            print(f"loading mesh_data from {remeshed_path}")
            mesh_data = np.load(remeshed_path, allow_pickle=True)
        else:
            mesh_data = None

        mesh_obj = {}
        for i, mesh in enumerate(builder.shape_geo_src):
            if isinstance(mesh, wp.sim.Mesh):
                if mesh_data is None or hash(mesh) != mesh_data.get(f"hash-{i}"):
                    # force remesh if any mesh hash does not match
                    mesh_data = None
                    mesh_obj[f"hash-{i}"] = hash(mesh)
                    mesh.remesh(visualize=False)
                    mesh_obj[f"vertices-{i}"] = mesh.vertices
                    mesh_obj[f"indices-{i}"] = mesh.indices
                    mesh_obj[f"mass-{i}"] = mesh.mass
                    mesh_obj[f"com-{i}"] = mesh.com
                    mesh_obj[f"I-{i}"] = mesh.I
                else:
                    mesh.vertices, mesh.indices = (
                        mesh_data[f"vertices-{i}"],
                        mesh_data[f"indices-{i}"],
                    )
                    mesh.mass, mesh.com, mesh.I = (
                        mesh_data[f"mass-{i}"],
                        mesh_data[f"com-{i}"],
                        mesh_data[f"I-{i}"],
                    )

        if mesh_data is None:
            print("saving remeshed data to ", remeshed_path)
            np.savez(remeshed_path, **mesh_obj)

    def create_articulation(self, builder, remesh=True):
        if not self.floating:
            self.object_joint = self.base_joint = add_joint(
                builder,
                pos=self.base_pos,
                ori=self.base_ori,
                joint_type=self.joint_type,
                stiffness=self.stiffness,
                damping=self.damping,
                body_name="object",  # self.object_name + "_body_joint",
            )
        if self.model_path and self.model_path.endswith(".stl"):
            from tcdm.envs import asset_abspath

            obj_stl_path = asset_abspath(self.model_path)
            mesh, _ = parse_mesh(obj_stl_path)
            geom_size = (self.scale, self.scale, self.scale)
            builder.add_shape_mesh(
                body=self.object_joint,
                pos=(0.0, 0.0, 0.0),  # in Z-up frame, transform applied already
                rot=(1.0, 0.0, 0.0, 0.0),
                mesh=mesh,
                scale=geom_size,
                density=self.density,
                ke=self.contact_ke,
                kd=self.contact_kd,
            )
        elif self.model_path and self.model_path.endswith(".urdf"):
            asset_dir = os.path.join(DMANIP_ENVS_PATH, "assets")
            wp.sim.parse_urdf(
                os.path.join(asset_dir, self.model_path),
                builder,
                xform=wp.transform(self.base_pos, self.base_ori),
                floating=self.floating,
                base_joint=self.base_joint,
                density=self.density,
                scale=self.scale,
                armature=1e-4,
                stiffness=self.stiffness,
                damping=self.damping,
                shape_ke=self.contact_ke,
                shape_kd=self.contact_kd,
                shape_kf=1.0e2,
                shape_mu=1.0,
                limit_ke=1.0e4,
                limit_kd=1.0e1,
                enable_self_collisions=True,
                parse_visuals_as_colliders=False,
                collapse_fixed_joints=True,
            )
        else:
            add_object(
                builder,
                self.object_joint,
                self.object_type,
                self.base_ori,
                self.density,
                self.contact_ke,
                self.contact_kd,
                scale=self.scale,
            )

        if remesh:
            try:
                self.remesh(builder)
            except Exception as e:
                print("remesh failed", e)


class OperableObjectModel(ObjectModel):
    def __init__(
        self,
        object_type,
        object_id=None,
        base_pos=(0.0, 0.075, 0.0),
        base_ori=(np.pi / 2, 0.0, 0.0),
        contact_ke=1.0e4,
        contact_kd=100.0,
        scale=0.4,
        density=1e2,
        stiffness=0.0,
        damping=0.0,
        model_path: str = "",
        base_joint=None,
        base_body_name="base",
        floating=False,
        use_mesh_extents=False,
        continuous_joint_type="screw",
        joint_limits=None,
    ):
        if isinstance(stiffness, int):
            stiffness = float(stiffness)
        if isinstance(damping, int):
            damping = float(damping)
        self.object_id = object_id
        super().__init__(
            object_type,
            base_pos=base_pos,
            base_ori=base_ori,
            contact_ke=contact_ke,
            contact_kd=contact_kd,
            scale=scale,
            stiffness=stiffness,
            damping=damping,
            density=density,
            base_body_name=base_body_name,
        )
        self.use_mesh_extents = use_mesh_extents
        self.floating = floating
        self.base_joint = base_joint
        self.density = density
        assert model_path and os.path.splitext(model_path)[1] == ".urdf"
        self.model_path = model_path
        self.continuous_joint_type = continuous_joint_type
        self.joint_limits = None  # joint_limits

    def get_object_extents(self):
        asset_dir = os.path.join(DMANIP_ENVS_PATH, "assets")
        obj_dir = os.path.split(self.model_path)[0]
        obj_filepath = os.path.join(asset_dir, obj_dir, "merged.obj")
        if os.path.split(obj_dir)[1].isdigit():
            obj_filepath = os.path.join(asset_dir, obj_dir, os.path.split(obj_dir)[1] + "_merged.obj")

        # TODO: fix getting correct obj path for all objects
        if not os.path.exists(obj_filepath):
            return

        obj_mesh = trimesh.load(obj_filepath)
        self.object_bounds = obj_mesh.extents * self.scale
        print("extents: ", self.object_bounds)

        if self.object_bounds[1] / 2 > self.base_pos[1]:
            print("object too tall, adjusting base position")
            self.base_pos = (
                self.base_pos[0],
                max(self.base_pos[1], self.object_bounds[1] / 2),
                self.base_pos[2],
            )

    def create_articulation(self, builder, remesh=True):
        asset_dir = os.path.join(DMANIP_ENVS_PATH, "assets")
        if self.use_mesh_extents:
            self.get_object_extents()
        joint_count_before = builder.joint_axis_count
        wp.sim.parse_partnet_urdf(
            os.path.join(asset_dir, self.model_path),
            builder,
            xform=wp.transform(self.base_pos, self.base_ori),
            floating=self.floating,
            base_joint=self.base_joint,
            density=self.density,
            scale=self.scale,
            armature=1e-4,
            stiffness=self.stiffness,
            damping=self.damping,
            shape_ke=self.contact_ke,
            shape_kd=self.contact_kd,
            shape_kf=1.0e2,
            shape_mu=1.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=True,
            parse_visuals_as_colliders=False,
            collapse_fixed_joints=True,
            continuous_joint_type=self.continuous_joint_type,
        )

        if remesh:
            try:
                self.remesh(builder)
            except ImportError as e:
                print("Skipped remeshing due to following error:", e)
                pass

        self.builder = builder
        if self.joint_limits is not None:
            joint_count_after = builder.joint_axis_count
            self.builder.joint_limit_lower[joint_count_before:joint_count_after] = self.joint_limits[0]
            self.builder.joint_limit_upper[joint_count_before:joint_count_after] = self.joint_limits[1]
        return builder


def object_generator(object_type, **kwargs):
    class __DexObj__(ObjectModel):
        def __init__(self, **override_kwargs):
            kwargs.update(override_kwargs)
            super().__init__(object_type=object_type, **kwargs)

    return __DexObj__


def operable_object_generator(object_type, **kwargs):
    class __OpDexObj__(OperableObjectModel):
        def __init__(self, **override_kwargs):
            kwargs.update(override_kwargs)
            super().__init__(object_type=object_type, **kwargs)

    return __OpDexObj__


def get_object_xform(object_type, object_id=None, **obj_kwargs):
    model = OBJ_MODELS.get(object_type)
    if object_id:
        model = model.get(object_id)(**obj_kwargs)
    else:
        model = model(**obj_kwargs)
    return model.base_pos, (0, 0, 0, 1)


StaplerObject = object_generator(ObjectType.TCDM_STAPLER, base_pos=(0.0, 0.01756801, 0.0), scale=1.3)
OctprismObject = object_generator(ObjectType.OCTPRISM, scale=1.0)

ReposeCubeObject = object_generator(
    ObjectType.REPOSE_CUBE,
    base_pos=(0.0, 0.35, 0.0),
    base_ori=(0.0, 0.0, 0.0),
    scale=1,
    density=1e2,
    stiffness=0.0,
    damping=0.0,
    contact_ke=1e3,  # shape_ke
    contact_kd=1e2,  # shape_kd, same as env_allegro.py
    floating=True,
    # base_joint="px, py, pz, rx, ry, rz",
    model_path="isaacgymenvs/objects/cube_multicolor.urdf",
)


SprayBottleObject = operable_object_generator(
    ObjectType.SPRAY_BOTTLE,
    base_pos=(0.0, 0.11, 0.0),
    base_ori=(0.0, -np.pi / 4, 0.0),
    scale=1.0,
    density=1.0,
    # base_joint=None,
    base_joint="rx, ry, rz",
    model_path="spray_bottle/mobility.urdf",
)

PillBottleObject = operable_object_generator(
    ObjectType.PILL_BOTTLE,
    # base_pos=(0.0, 0.18, 0.0),
    use_mesh_extents=True,
    base_ori=(-np.pi / 2, 0.0, 0.0),
    scale=0.14,
    stiffness=[11, 11],
    damping=[0.0, 0.0],
    # base_ori=(np.pi / 17, 0.0, 0.0),
    model_path="pill_bottle/mobility.urdf",
    continuous_joint_type="revolute",
)

bottle_ids = ("3558", "3574", "3616")
BottleObjects = {
    bottle_id: operable_object_generator(
        ObjectType.BOTTLE,
        object_id=bottle_id,
        # base_pos=(0.0, 0.01756801, 0.0),
        base_ori=(-0.5 * np.pi, 0.0, 0.0),
        use_mesh_extents=True,
        scale=0.4,
        # base_ori=(np.pi / 17, 0.0, 0.0),
        stiffness=[12, 12],
        damping=[2.0, 0.2],
        model_path=f"Bottle/{bottle_id}/mobility.urdf",
    )
    for bottle_id in bottle_ids
}

dispenser_ids = ("101539", "101417", "101517", "101540", "103405", "103619")
DispenserObjects = {
    dispenser_id: operable_object_generator(
        ObjectType.DISPENSER,
        object_id=dispenser_id,
        base_pos=(0.0, 0.01756801, 0.0),
        base_ori=(-0.5 * np.pi, 0.0, 0.0),
        scale=0.4,
        stiffness=[10.0, 10.0],
        damping=[0.5, 0.5],
        use_mesh_extents=True,
        # base_ori=(np.pi / 17, 0.0, 0.0),
        model_path=f"Dispenser/{dispenser_id}/mobility.urdf",
        continuous_joint_type="revolute",
    )
    for dispenser_id in dispenser_ids
}


SoapDispenserObject = operable_object_generator(
    ObjectType.SOAP_DISPENSER,
    base_pos=(0.0, 0.01756801, 0.0),
    base_ori=(-0.5 * np.pi, 0.0, 0.0),
    scale=0.4,
    use_mesh_extents=True,
    # base_ori=(np.pi / 17, 0.0, 0.0),
    stiffness=[100, 100],
    damping=[2.0, 0.2],
    model_path="Dispenser/101490/mobility.urdf",
)

eyeglasses_ids = ("101284", "102586", "103189")
EyeglassesObjects = {
    eyeglasses_id: operable_object_generator(
        ObjectType.EYEGLASSES,
        object_id=eyeglasses_id,
        base_pos=(0.0, 0.01756801, 0.0),
        base_ori=(-np.pi / 2, 0.0, 0.0),
        scale=0.2,
        use_mesh_extents=True,
        # base_ori=(np.pi / 17, 0.0, 0.0),
        model_path=f"Eyeglasses/{eyeglasses_id}/mobility.urdf",
    )
    for eyeglasses_id in eyeglasses_ids
}

faucet_ids = ("152", "1556", "156", "2170")
FaucetObjects = {
    faucet_id: operable_object_generator(
        ObjectType.FAUCET,
        object_id=faucet_id,
        base_pos=(0.0, 0.01756801, 0.0),
        base_ori=(-np.pi / 2, 0.0, 0.0),
        scale=0.4,
        # base_ori=(np.pi / 17, 0.0, 0.0),
        model_path=f"Faucet/{faucet_id}/mobility.urdf",
    )
    for faucet_id in faucet_ids
}

pliers_ids = ("100142", "100182", "100705", "102074")
PliersObjects = {
    pliers_id: operable_object_generator(
        ObjectType.PLIERS,
        object_id=pliers_id,
        base_pos=(0.0, 0.01756801, 0.0),
        base_ori=(-np.pi / 2, 0.0, 0.0),
        scale=0.24,
        # base_ori=(np.pi / 17, 0.0, 0.0),
        model_path=f"Pliers/{pliers_id}/mobility.urdf",
    )
    for pliers_id in pliers_ids
}

scissors_ids = ("10449", "10889", "11080", "11100")
ScissorsObjects = {
    scissors_id: operable_object_generator(
        ObjectType.SCISSORS,
        object_id=scissors_id,
        base_pos=(0.0, 0.01756801, 0.0),
        base_ori=(-np.pi / 2, 0.0, 0.0),
        scale=0.2,
        # base_ori=(np.pi / 17, 0.0, 0.0),
        base_joint="px, py, pz",
        model_path=f"Scissors/{scissors_id}/mobility.urdf",
    )
    for scissors_id in scissors_ids
}

stapler_ids = ("102990", "103271", "103792")
StaplerObjects = {
    stapler_id: operable_object_generator(
        ObjectType.STAPLER,
        object_id=stapler_id,
        base_pos=(0.0, 0.01756801, 0.0),
        base_ori=(0, 0.0, 0.0),
        scale=0.14,
        # base_ori=(np.pi / 17, 0.0, 0.0),
        model_path=f"Stapler/{stapler_id}/mobility.urdf",
    )
    for stapler_id in stapler_ids
}

switch_ids = ("100866", "100901", "102812")  # "100883"
SwitchObjects = {
    switch_id: operable_object_generator(
        ObjectType.SWITCH,
        object_id=switch_id,
        base_pos=(0.0, 0.15, 0.0),
        base_ori=(0.0, 0.0, 0.0),
        scale=0.1,
        base_joint=None,
        # base_ori=(np.pi / 17, 0.0, 0.0),
        model_path=f"Switch/{switch_id}/mobility.urdf",
    )
    for switch_id in switch_ids
}

usb_ids = ("100061", "100065", "100109")  # , "102052")
USBObjects = {
    usb_id: operable_object_generator(
        ObjectType.USB,
        object_id=usb_id,
        base_pos=(0.0, 0.01756801, 0.0),
        base_ori=(-np.pi / 2, 0.0, 0.0),
        scale=0.4,
        # base_ori=(np.pi / 17, 0.0, 0.0),
        use_mesh_extents=True,
        model_path=f"USB/{usb_id}/mobility.urdf",
    )
    for usb_id in usb_ids
}


OBJ_MODELS = {}
OBJ_NUM_JOINTS = {}
OBJ_MODELS[ObjectType.TCDM_STAPLER] = StaplerObject
OBJ_NUM_JOINTS[ObjectType.TCDM_STAPLER] = 7
OBJ_MODELS[ObjectType.OCTPRISM] = OctprismObject
OBJ_NUM_JOINTS[ObjectType.OCTPRISM] = 7
OBJ_MODELS[ObjectType.SPRAY_BOTTLE] = SprayBottleObject
OBJ_NUM_JOINTS[ObjectType.SPRAY_BOTTLE] = 1
OBJ_MODELS[ObjectType.PILL_BOTTLE] = PillBottleObject
OBJ_NUM_JOINTS[ObjectType.PILL_BOTTLE] = 2
OBJ_MODELS[ObjectType.BOTTLE] = BottleObjects
OBJ_NUM_JOINTS[ObjectType.BOTTLE] = 1
OBJ_MODELS[ObjectType.DISPENSER] = DispenserObjects
OBJ_NUM_JOINTS[ObjectType.DISPENSER] = 2

OBJ_MODELS[ObjectType.SOAP_DISPENSER] = SoapDispenserObject
OBJ_NUM_JOINTS[ObjectType.SOAP_DISPENSER] = 1


OBJ_MODELS[ObjectType.EYEGLASSES] = EyeglassesObjects
OBJ_NUM_JOINTS[ObjectType.EYEGLASSES] = 2
OBJ_MODELS[ObjectType.FAUCET] = FaucetObjects
OBJ_NUM_JOINTS[ObjectType.FAUCET] = 2
OBJ_MODELS[ObjectType.PLIERS] = PliersObjects
OBJ_NUM_JOINTS[ObjectType.PLIERS] = 1
OBJ_MODELS[ObjectType.SCISSORS] = ScissorsObjects
OBJ_NUM_JOINTS[ObjectType.SCISSORS] = 1
OBJ_MODELS[ObjectType.STAPLER] = StaplerObjects
OBJ_NUM_JOINTS[ObjectType.STAPLER] = 2
OBJ_MODELS[ObjectType.SWITCH] = SwitchObjects
OBJ_NUM_JOINTS[ObjectType.SWITCH] = 1
OBJ_MODELS[ObjectType.USB] = USBObjects
OBJ_NUM_JOINTS[ObjectType.USB] = 1
OBJ_MODELS[ObjectType.REPOSE_CUBE] = ReposeCubeObject
OBJ_NUM_JOINTS[ObjectType.REPOSE_CUBE] = 6


def get_num_acts(object_type):
    return OBJ_NUM_JOINTS.get(object_type, 1)
