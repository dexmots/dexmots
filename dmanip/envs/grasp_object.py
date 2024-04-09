# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import warp as wp
import os
os.environ['DISPLAY'] = ':1'
import warp.sim
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json

from scipy.spatial.transform import Rotation as R
from localscope import localscope
from dmanip.utils import builder as bu
from dmanip.envs import HandObjectTask, ObjectTask
from dmanip.utils.rewards import l1_dist
from dmanip.utils import torch_utils as tu
from dmanip.utils.common import HandType, ObjectType, ActionType, GraspParams, load_grasps_npy, run_env
from pyglet.math import Vec3 as PyVec3

wp.init()

# %% [markdown]
# # Load grasps

# %%
hand_type = HandType["ALLEGRO"]


# %%
@localscope.mfc
def get_object_type_id(object_code):
    object_type = "_".join(filter(lambda x: not( x == "merged" or x.isdigit()), object_code.split('_')))
    object_id = object_code.rstrip('_merged').split('_')[-1]
    if not object_id.isdigit():
        object_id = None
    return object_type, object_id


allegro_grasps  = "./graspdata/allegro"
shadow_grasps = "./graspdata/shadow"
# print("allegro grasps:", osp.listdir(f"{allegro_grasps}/"))
# # !ls {allegro_grasps}
# print("shadow grasps:", osp.listdir(f"{shadow_grasps}/"))
# # !ls {shadow_grasps}

grasps = json.loads(open("./graspdata/good_grasps.txt").read())


# Constructs the grasp_dicts with loaded grasp_params from each grasp in good grasps
for grasp_dict in grasps['grasps']:
    object_code = grasp_dict['object_code']
    object_type, object_id = get_object_type_id(object_code)
    try:
        ot = ObjectType[object_type.upper()]
    except:
        continue
    grasp_dict.update(dict(object_type=object_type, object_id=object_id))
    grasp_npy = os.path.join(allegro_grasps,f"{object_code}.npy")
    grasp_dict['grasp_file'] = grasp_npy
    grasp_params = map(lambda x: x[1], filter(lambda x: x[0] in grasp_dict['grasp_ids'], enumerate(load_grasps_npy(ot, object_id, hand_type))))
    grasp_dict['params'] = list(grasp_params)

# %%
# object_code = "Stapler_102990_merged"
object_codes = list(map(lambda x: x.rstrip(".npy"), filter(lambda x: 'Stapler' in x, os.listdir(allegro_grasps))))
for object_code in object_codes:

    g = [g for g in grasps['grasps'] if g['object_code'] == object_code][0]

    object_type = ObjectType[g['object_type'].upper()]
    object_id = g.get("object_id", None)
    object_model = bu.OBJ_MODELS.get(object_type)
    if object_id:
        object_model = object_model.get(object_id)

    object_model = object_model(use_mesh_extents=True)
    object_model.get_object_extents()
    print("object_code", object_code, "base_pos", object_model.base_pos)

# %% [markdown]
# ## Load in grasp for desired object type

# %%
g = [g for g in grasps['grasps'] if g['object_id'] == "102990"][0]
object_type = ObjectType[g['object_type'].upper()]
object_id = g.get("object_id", None)
g

# %% [markdown]
# ## Test Object joints/start pose in environment
# ``` python
# object_env = ObjectTask(10, object_type=object_type, object_id=object_id)
# object_env.reset()
# object_env.render()
#
# lim_upper = object_env.model.joint_limit_upper.numpy().reshape(object_env.num_envs, -1)
# lim_lower = object_env.model.joint_limit_lower.numpy().reshape(object_env.num_envs, -1)
# print(lim_lower, lim_upper)
# object_env.start_joint_q = tu.to_torch(lim_upper) # tu.to_torch((lim_upper-lim_lower)/7+lim_lower)
# # object_env.start_joint_q[:, 0] = tu.to_torch(lim_upper[:, 0]) #  + (lim_upper[:, 1]-lim_lower[:,1])/64)
#
# print(object_env.start_joint_q)
#
# object_env.reset()
# # object_env.load_camera_params("camera_params_closeup.npz")
# object_env.render()
#
# import time
#
# act = tu.unscale(object_env.start_joint_q, object_env.action_bounds[0], object_env.action_bounds[1])
# for _ in range(60*60):
#     time.sleep(1/60)
#     object_env.step(act)
# ```

# %% [markdown]
# ## Create HandObjectEnv
# sets grasp pose to what is contained in grasps dict

# %%
object_model = bu.OBJ_MODELS.get(object_type)['102990']()
print(object_model.base_pos, object_model.base_pos, object_model.base_joint)

# %%
from dmanip.envs.environment import RenderMode

gparams = g['params']
rew_params = {"hand_joint_pos_err": (l1_dist, ("target_qpos", "hand_qpos"), 1.0)}

object_model = bu.OBJ_MODELS.get(object_type)
if object_id:
    object_model = object_model.get(object_id)

object_model = object_model(use_mesh_extents=True)

# loads env with grasps for desired object_code
env = HandObjectTask(1, 1, episode_length=1000, object_type=object_type, object_id=object_id,
                     stochastic_init=False, reward_params=rew_params, load_grasps=True,
                     grasp_id=None, hand_start_position=(0,0,0),
                     hand_start_orientation=(0,0,0), action_type=ActionType["POSITION"], 
                     fix_position=True, fix_orientation=True, render_mode=RenderMode["USD"])

env.reset()

# %%
env.body_q

# %%
env.stochastic_init = True
env.reset()

# %%
hand_base_xform

# %%
# hand_base_xform = env.model.joint_X_p.numpy()[0]
grasp_xform = tu.to_torch(env.grasp.xform[None])
object_xform = tu.to_torch(env.model.joint_X_p.numpy()[env.joint_name_to_idx['base_joint']][0:1])

# object_base_xform = np.concatenate([env.object_model.base_pos, env.object_model.base_ori])
tu.tf_combine(object_xform[:,3:], object_xform[:, :3], grasp_xform[:, 3:], grasp_xform[:, :3])

# %%
env.model.shape_count

# %%
env.model.shape_body.numpy()

# %%
object_base_xform

# %%
env.joint_name_to_idx['base_joint']

# %%

# %%
env.joint_q[-6:]

# %%
env.object_model.base_joint

# %%
xforms = np.stack([hand_base_xform, grasp_xform, object_base_xform])

# %%
from scipy.spatial.transform import Rotation as R


# %%
def get_origin_lines(xform=None, name=""):
    if xform is None:
        rot = np.eye(3)
        offset = np.zeros(3)
    else:
        rot = R.from_quat(xform[3:]).as_matrix()
        offset = xform[:3]

    x_lines = np.array([[0.0, 0.0, 0.0], rot @ [0.1, 0.0, 0.0]])
    y_lines = np.array([[0.0, 0.0, 0.0], rot @ [0.0, 0.1, 0.0]])
    z_lines = np.array([[0.0, 0.0, 0.0], rot @ [0.0, 0.0, 0.1]])
    x_lines += offset
    y_lines += offset
    z_lines += offset
    lines = [
        go.Scatter3d(
            x=x_lines[:, 0],
            y=x_lines[:, 1],
            z=x_lines[:, 2],
            mode="lines",
            line=dict(width=2, color="red"),
            name=f"{name} Coord X Axis",
        ),
        go.Scatter3d(
            x=y_lines[:, 0],
            y=y_lines[:, 1],
            z=y_lines[:, 2],
            mode="lines",
            line=dict(width=2, color="green"),
            name=f"{name} Coord Y Axis",
        ),
        go.Scatter3d(
            x=z_lines[:, 0],
            y=z_lines[:, 1],
            z=z_lines[:, 2],
            mode="lines",
            line=dict(width=2, color="blue"),
            name=f"{name} Coord Z Axis",
        ),
    ]
    return lines


# %%
from plotly import graph_objects as go

fig = go.Figure()
fig.add_traces(get_origin_lines(hand_base_xform, "Hand"))
fig.add_traces(get_origin_lines(grasp_xform, "Grasp"))
fig.add_traces(get_origin_lines(object_base_xform, "Object"))
fig.show()

# %%
ori, pos = tu.tf_combine(tu.to_torch((0,0,0,1.)), tu.to_torch(env.object_model.base_pos), 
              tu.to_torch(env.grasp.xform[3:]), tu.to_torch(env.grasp.xform[:3]))
pos, ori

# %%
env.model.joint_X_p.numpy()[0:3]

# %%
torch.tensor(env.hand_start_position), torch.tensor(env.hand_start_orientation)

# %%
env.model.joint_target_ke.numpy()

# %%
import time

act = tu.unscale(env.start_joint_q[:, env.env_joint_target_indices], env.action_bounds[0], env.action_bounds[1])
for _ in range(100):
    env.step(act)

env.renderer.save()

# %%
from pprint import pprint

print("Body keys:")
pprint(list(env.body_name_to_idx.keys()))
print("Joint keys:")
pprint(list(env.joint_name_to_idx.keys()))

# %%
env.model.joint_q_start.numpy()[16:19]

# %%
start_idx = env.joint_name_to_idx["base_joint"].item()
env.joint_q[env.model.joint_q_start.numpy()[start_idx]:env.model.joint_q_start.numpy()[start_idx+1]]

# %%
env.object_model.base_pos

# %%
env.object_model.base_ori

# %%
env.model.joint_X_p.numpy()[16:]

# %%
env.model.joint_X_p.numpy()[env.model.joint_parent.numpy() == -1]

# %%
np.array(env.model.joint_name)[env.model.joint_parent.numpy() == -1]

# %%
env.body_q[env.body_name_to_idx["base"]]

# %%
print(env.model.joint_X_p)

# %%
obs_dict = env.extras['obs_dict']
obs_dict['goal_joint_pos']

# %%
[(k, v.shape) for k, v in env.extras['obs_dict'].items()]

# %%
env._set_hand_base_xform()
new_state = env.model.state()
wp.sim.eval_fk(env.model, env._joint_q, env._joint_qd, env.state())

# %%
env.sample_grasps()

# %%
while True: env.render()

# %%
env.start_joint_q[:, env.env_joint_target_indices].shape

# %%
env.renderer.save()


# %%
def get_base_links(link_pattern):
    base_links = np.array(list(map(
                lambda x: x[0], 
                filter(
                    lambda x: link_pattern in x[1], 
                    enumerate(env.model.body_name)))))
    return base_links

hand_base_links = get_base_links('link_0')
hand_base_pos = env.body_q[base_links].view(5, -1, 7).mean(dim=1)[:, :3] - env.env_offsets
obj_base_links = get_base_links('base')
obj_base_pos = env.body_q[obj_base_links].view(5, 7)[:, :3] - env.env_offsets

obj_base_pos - hand_base_pos

# %%
grasp_ori = env.grasp.xform[3:]
grasp_pos = env.grasp.xform[:3]
pos = tu.tf_apply(*[tu.to_torch(x) for x in [grasp_ori, obj_base_pos[0], grasp_pos]])

# %%
env.model.joint_X_p.numpy()[0, :3] - env.env_offsets[0].cpu().numpy()

# %%
hand_base_pos

# %%
pos

# %%
env.hand_start_position

# %%
env.grasp.xform[:3]

# %%
import ipywidgets as widgets
import time
import torch
from IPython.display import display


class MySliderApp:
    def __init__(self, initial_vals, env_step, env_render, num_envs):
        self.sliders = []
        for i, val in enumerate(initial_vals):
            s = widgets.FloatSlider(
                value=val,
                min=-1.0,
                max=1.0,
                step=0.01,
                description=f"Joint DoF {i+1}",
                continuous_update=False
            )
            s.observe(self.on_slider_change, names=["value"])
            self.sliders.append(s)
        self.env_step = env_step
        self.env_render = env_render
        self.num_envs = num_envs
        self.output = widgets.Output()
        self.loaded_once = False  # Initialize flag
        self.display_widgets()

    def display_widgets(self):
        display(widgets.VBox([*self.sliders, self.output]))

    def on_slider_change(self, event):
        with self.output:
            actions = torch.as_tensor([s.value for s in self.sliders]).view(1, -1).repeat(self.num_envs, 1)
            self.env_step(actions)

            # If render_while_waiting has not been called before, call it now and set flag to True
            if not self.loaded_once:
                self.render_while_waiting()
                self.loaded_once = True

    def render_while_waiting(self):
        wait_time = 1  # Set wait time in seconds
        while not self.update_fn(False):  # While update function is not running
            time.sleep(wait_time)
            self.env_render()  # Call render function
            print("Rendering...")

initial_vals = [0.0, 0.0, 0.0]  # You can change this to set initial slider values

def render():
    print("Rendering...")

app = MySliderApp(initial_vals, env.step, env.render, env.num_envs)

# env.grasp_joint_q = None
# env.grasps = gparams
# pi = lambda x, t: tu.unscale(env.hand_init_q, env.action_bounds[0], env.action_bounds[1])
# run_env(env, pi, num_steps=1000)

# %%
(lower <= env.grasp.joint_pos) ^ (env.grasp.joint_pos <= upper)

# %%
lower = env.model.joint_limit_lower.numpy().reshape(env.num_envs, -1)[0, :16]
upper = env.model.joint_limit_upper.numpy().reshape(env.num_envs, -1)[0, :16]

# %%
env.model.joint_limit_upper.numpy().reshape(env.num_envs, -1)[0]

# %%
tu.unscale(env.hand_init_q, env.action_bounds[0], env.action_bounds[1]).min()

# %%
tu.unscale(env.hand_init_q, env.action_bounds[0], env.action_bounds[1]).min()

# %%
env.assign_actions(tu.unscale(env.hand_init_q, env.action_bounds[0], env.action_bounds[1]))

# %%

# %%
env.model.joint_target.numpy().reshape(env.num_envs,-1)[:, :16] - env.hand_init_q.cpu().numpy()

# %%
env.render()

# %%
env.load_camera_params()
plt.imshow(env.render("rgb_array"))


# %%
@localscope.mfc
def plot_grasp_xform(grasp: GraspParams):
    # Original quaternion
    pos = grasp.xform[:3]
    quat = grasp.xform[3:]

    # Rotation matrix
    rot_mat = R.from_quat(quat).as_matrix()

    # Axes vectors
    # x_axis = rot_mat[:, 0]
    # y_axis = rot_mat[:, 1]
    # z_axis = rot_mat[:, 2]

    x_axis = np.array([0.533986, -0.627104, 0.56728])
    y_axis = np.array([0.779297, 0.595503, -0.1979])
    z_axis = np.array([-0.324763, 0.502729, 0.801218])

    # Create a 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=[pos[0], quat[0]],
            y=[pos[1], quat[1]],
            z=[pos[2], quat[2]],
            name='Original Quaternion',
            marker=dict(size=5, color='orange')
        ),
        go.Scatter3d(
            x=[pos[0], x_axis[0]],
            y=[pos[1], x_axis[1]],
            z=[pos[2], x_axis[2]],
            name='X-Axis',
            marker=dict(size=5, color='green')
        ),
        go.Scatter3d(
            x=[pos[0], y_axis[0]],
            y=[pos[1], y_axis[1]],
            z=[pos[2], y_axis[2]],
            name='Y-Axis',
            marker=dict(size=5, color='blue')
        ),
        go.Scatter3d(
            x=[pos[0], z_axis[0]],
            y=[pos[1], z_axis[1]],
            z=[pos[2], z_axis[2]],
            name='Z-Axis',
            marker=dict(size=5, color='red')
        ),
        go.Scatter3d(
            x=[pos[0], 1],
            y=[pos[1], 0],
            z=[pos[2], 0],
            name='Original X-Axis',
            marker=dict(size=2, color='green')
        ),
        go.Scatter3d(
            x=[pos[0], 0],
            y=[pos[1], 1],
            z=[pos[2], 0],
            name='Original Y-Axis',
            marker=dict(size=2, color='blue')
        ),
        go.Scatter3d(
            x=[pos[0], 0],
            y=[pos[1], 0],
            z=[pos[2], 1],
            name='Original Z-Axis',
            marker=dict(size=2, color='red')
        ),
    ])

    # Set the layout
    fig.update_layout(
        title="Quaternion and Axes Vectors",
        scene=dict(
            xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    ),
    showlegend=True
    )

    # Show the plot
    fig.show()

# %%
env.calculateObservations().shape

# %%
plot_grasp_xform(grasps[0])

# %%
env.render()
env.renderer._camera_pos = PyVec3(0.0, .5, 2.0)
env.renderer.update_view_matrix()
env.render()

# %%
env.renderer.

# %%
pos, q = grasps[0].xform[:3], grasps[0].xform[3:]

# %%
# %matplotlib widget

# %%
while True:
    env.renderer.axis_instancer.allocate_instances(
        positions=[pos],
        rotations=,
        colors1=,
        colors2=)
    env.render()

# %%
env.env_joint_mask

# %%
builder = wp.sim.ModelBuilder()
bu.create_allegro_hand(
    builder,
    ActionType.POSITION,
    stiffness=1000,
    damping=0.1,
    base_joint=None,
    floating_base=True,
    # base_joint="rx, ry, rz",
    hand_start_position=(0,0.4,0.),
    hand_start_orientation=(0.,0.,0.),
)

# %%
model = builder.finalize()

# %%
np.concatenate([grasps[0].xform, grasps[0].joint_pos]).size

# %%
state = model.state()
state.joint_q.assign(np.concatenate([grasps[0].xform, grasps[0].joint_pos]))

# %%
print(state.joint_q)

# %%
state = model.state()

wp.sim.eval_fk(model, state.joint_q, state.joint_qd, None, state)

# %%
renderer = wp.sim.render.SimRendererOpenGL(
    model,
    "env",
    up_axis="y",
    show_joints=True
)

# %%
renderer.show_joints= False
renderer.draw_axis = False

# %%
while True:
    renderer.begin_frame(0.0)
    renderer.render(state)
    renderer.end_frame()

# %%
print(state.joint_q)

# %%
HandObjectTask.usd_render_settings['show_joints'] = True
env = HandObjectTask(1,1,100, hand_type=HandType["SHADOW"])

obs = env.reset()
env.render()
env.renderer._camera_pos = PyVec3(0.0, .5, 2.0)
env.renderer.update_view_matrix()
env.render()

# %%
while True:
    env.render()

# %%
from common import load_grasps_npy

# %%
grasps = load_grasps_npy("/scr-ssd/ksrini/tyler-DexGraspNet/grasp_generation/graspdata_allegro/spray_bottle_merged.npy")

# %%
env.reset_buf != 0

# %%
env.hand_joint_start = 0

# %%
env.model.body_name

# %%
env.model.joint_X_p

# %%
env._set_hand_base_xform(env.reset_buf != 0, torch.tensor(grasps[0].xform, dtype=torch.float32, device="cuda"))

# %%
from rewards import l1_dist

# %%
env.rew_params = {"hand_joint_pos_err": (l1_dist, ("target_qpos", "hand_qpos"), 1.0)}

# %%
env.renderer.show_joints = True
env.render()

# %%
env.step(torch.tensor(env.action_space.sample()*0, dtype=torch.float32, device='cuda'))

# %%
env.rew_params

# %%
env.render()
