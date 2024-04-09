import numpy as np
import os
import io
import math
import os.path as osp
import matplotlib.pyplot as plt
import plotly
import pickle
import plotly.graph_objects as go
import warp as wp
import warp.sim.render

from dmanip.utils.common import gen_unique_filename, to_numpy
from matplotlib.animation import FuncAnimation
from typing import Optional
from tqdm import tqdm
from PIL import Image


def create_animation(
    line_data: np.ndarray,  # array of shape [n_frames, T, n_lines]
    ref_line: Optional[np.ndarray],  # array of shape [T, n_lines]
    save_dir: str,
    name: str,
    xlabel=None,
    ylabel=None,
    title=None,
    ylim=None,
):
    num_frames = line_data.shape[0]
    if len(line_data.shape) == 2:
        line_data = line_data[:, :, None]  # add empty dim
    assert num_frames > 1
    fig, ax = plt.subplots()
    lines = ax.plot(line_data[0])
    if ref_line is not None:
        line2 = ax.plot(ref_line, color="red")[0]

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    def update(frame):
        for i, line in enumerate(lines):
            line.set_ydata(line_data[frame + 1, :, i])
        return lines

    anim = FuncAnimation(fig, update, frames=num_frames - 1, interval=100)
    save_path = gen_unique_filename(osp.join(save_dir, name), max_id=20)
    anim.save(save_path, writer="pillow")


def plot_states(save_dir, ref, states, color=None):
    fig, ax = plt.subplots()
    ax.plot(ref, label="ref", color="red")
    # print("states", np_states)
    ax.plot(states, label="states", color=color)
    ax.legend()
    plt.savefig(f"{save_dir}/traj_states.png")
    plt.close(fig)


def plot_logs(log, save_dir):
    valid_keys = [k for k in log if len(log[k]) > 0]
    # iterate plot path
    traj_plot_path = gen_unique_filename(osp.join(save_dir, "traj_plot.png"), max_id=20)

    nrows, ncols = math.ceil(len(valid_keys) / 3), min(len(valid_keys), 3)
    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )
    for i, k in enumerate(valid_keys):
        axes = ax[i // ncols][i % ncols]
        data = log[k]
        if "joint pos" in k:
            data = np.array(data) / np.pi
            axes.set_ylabel("joint pos (pi radians)")
        axes.plot(data)
        axes.set_title(k)
        if k == "losses":
            axes.set_yscale("log")

    plt.savefig(traj_plot_path)
    plt.close(fig)


## Helper functions


def _make_dir(filename):
    folder = os.path.dirname(filename)
    os.makedirs(folder, exist_ok=True)


def save_video(video_frames, filename, fps=10, video_format="mp4"):
    import skvideo.io

    if len(video_frames) == 0:
        return False

    assert fps == int(fps), fps
    _make_dir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            "-r": str(int(fps)),
        },
        outputdict={"-f": video_format, "-pix_fmt": "yuv420p"},
    )

    return True


def generate_sample_data():
    log_dir = os.path.join(os.getcwd(), "sample")  # log directory
    os.makedirs(log_dir, exist_ok=True)

    num_steps = 50
    x_list, y_list = np.linspace(0, 1, num_steps + 1), np.linspace(0, 1, num_steps + 1)
    xx, yy = np.meshgrid(x_list, y_list)  # x and y list
    ll = xx**2 + yy + 1  # loss values with shape (num_step + 1, num_step + 1)
    grads = np.stack(
        [2 * xx, np.ones_like(yy), np.zeros_like(yy)], axis=-1
    )  # grad values with shape (num_steps + 1, num_steps + 1, 3)

    np.savez(os.path.join(log_dir, "info.npz"), xx=xx, yy=yy, ll=ll, grad=grads)

    return log_dir, num_steps


## Plotting script


def plot_space_2d(
    log_dir,
    base_params=[0.0, 0.0],
    dim_ids=[0, 1],
    xlim=(0.1, 0.6),
    ylim=(0.1, 0.5),
    num_steps=50,
    xlabel="Initial Velocity",
    ylabel="Cloth Length",
    history=[],
    include_plotly=True,
    return_image=False,
    levels=60,
    requires_grad=False,
    render_video=False,
):
    plot_args = locals()
    with open(os.path.join(log_dir, "info_meta.p"), "wb") as f:
        pickle.dump(plot_args, f)
    plotting_dir = os.path.join(log_dir, "plotting2d")
    os.makedirs(plotting_dir, exist_ok=True)

    info_path = os.path.join(log_dir, "info.npz")
    xx_raw = np.linspace(xlim[0], xlim[1], num_steps + 1)
    yy_raw = np.linspace(ylim[0], ylim[1], num_steps + 1)
    if os.path.isfile(info_path):
        data = np.load(info_path)
        xx, yy, ll = data["xx"], data["yy"], data["ll"]
        if "grad" in data.keys():
            grads = data["grad"]
        else:
            grads = None
    else:
        raise ValueError("Data file not found!")

    filename = os.path.join(log_dir, f"space2d.pdf")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    cs = plt.contourf(xx, yy, ll, cmap="RdGy", levels=levels)
    if requires_grad and grads is not None:
        scale_factor = max(1, num_steps // 20)
        xx_grad, yy_grad = np.meshgrid(xx_raw[::scale_factor], yy_raw[::scale_factor])
        grads = grads.reshape(len(xx_raw), len(yy_raw), len(base_params))
        grads = grads[::scale_factor, ::scale_factor].reshape(
            tuple(xx_grad.shape) + (len(base_params),)
        )
        grads = grads[..., dim_ids]
        grad_norms = np.linalg.norm(grads, axis=-1)
        non_zero_ix = grad_norms > 1e-6
        grads[non_zero_ix] /= grad_norms[non_zero_ix][:, None]
        grads[grad_norms <= 1e-6] *= 0
        plt.quiver(
            xx_grad,
            yy_grad,
            -grads[..., 0],
            -grads[..., 1],
            width=0.004,
            headlength=4,
            headaxislength=4,
            headwidth=4,
            pivot="mid",
            scale=27,
            scale_units="height",
        )
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel("Loss")
    if len(history) > 0:
        colors = [
            "#069af3",
            "#04d8b2",
            "#e6daa6",
            "#7e1e9c",
            "#ff81c0",
            "#fc5a50",
            "#fac205",
            "#000080",
            "#ffb4a2",
            "#588157",
        ]
        d0, d1 = dim_ids[0], dim_ids[1]
        if len(history.shape) == 2:
            ax.plot(history[:, d0], history[:, d1], "o--", lw=1, color="#069af3")
            ax.plot(history[[-1], d0], history[[-1], d1], "D", color="#069af3")
        else:
            for k in range(len(history)):
                ax.plot(
                    history[k, :, d0],
                    history[k, :, d1],
                    "o--",
                    lw=1,
                    color=colors[k % len(colors)],
                )
                ax.plot(
                    history[k, [-1], d0],
                    history[k, [-1], d1],
                    "D",
                    color=colors[k % len(colors)],
                )

    plt.tight_layout()
    if return_image:
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    else:
        plt.savefig(filename)
    plt.close(fig)

    # plotting 2d plot
    if include_plotly:
        plot_info = {
            "data": go.Surface(z=ll, x=xx_raw, y=yy_raw),
            "layout": {
                "width": 800,
                "height": 800,
                "scene": {
                    "xaxis": {"title": xlabel, "range": list(xlim)},
                    "yaxis": {"title": ylabel, "range": list(ylim)},
                    "zaxis": {"title": "Loss"},
                    "aspectratio": {"x": 1, "y": 1, "z": 1},
                },
            },
        }
        fig = go.Figure(plot_info)
        html_save_path = os.path.join(log_dir, "space2d.html")
        plotly.io.write_html(fig, file=html_save_path)

        # save video of rotating plot
        if render_video:
            T = 200
            images = []
            fig.update_layout(
                width=1200,
                height=1200,
                font=dict(size=15),
                margin=dict(t=20, r=40, l=40, b=40),
            )
            for t in tqdm(range(T), desc="Render space 2d"):
                cam_x = 1.5 * np.sqrt(2) * np.cos(np.pi * 2 / T * t + np.pi / 4)
                cam_y = 1.5 * np.sqrt(2) * np.sin(np.pi * 2 / T * t + np.pi / 4)
                camera = dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=cam_x, y=cam_y, z=1.2),
                )
                fig.update_layout(scene_camera=camera)
                fig_bytes = fig.to_image(format="png")
                buf = io.BytesIO(fig_bytes)
                img = Image.open(buf)
                images.append(np.asarray(img))
            save_video(images, os.path.join(log_dir, "space_2d.mp4"), fps=30)

    if return_image:
        return data


def render_line(
    renderer: wp.sim.render.SimRenderer,
    start_pt: wp.vec3,
    rot: wp.quat,
    length: float,
    color: tuple,
    name: str,
):
    end_pt = wp.vec3(length, 0.0, 0.0)
    X_w = wp.transform(wp.vec3(), rot)
    end_pt = wp.transform_point(X_w, end_pt)
    vertices = [tuple(start_pt), tuple(end_pt)]
    renderer.render_line_strip(name=name, vertices=vertices, color=color)


def render_fingertips(env, scale=0.1):
    points = np.concatenate(
        [to_numpy(env.extras[x]) for x in env.extras if "fingertip_pos" in x]
    )
    env.stage.render_points("debug_points", points, scale)


if __name__ == "__main__":
    log_dir, num_steps = generate_sample_data()
    plot_space_2d(
        log_dir,  # log directory where the info.npz file is found
        base_params=[
            0.0,
            0.0,
            0.0,
        ],  # base params that has same dimensionality as grads
        dim_ids=[0, 1],  # dimensions being plotted
        xlim=(0, 1),
        ylim=(0, 1),  # limits of x and y values being plotted
        num_steps=num_steps,  # number of values plotted along x and y
        xlabel="X Label",
        ylabel="Y Label",  # x and y labels
        levels=60,  # number of levels in the height map being plotted in 2d
        requires_grad=True,  # whether to plot gradients in the 2d plot
        include_plotly=True,  # whether to include a 3d plot in addition to the 2d one
        render_video=False,  # whether to also render a rotating 3d plot as a video
    )
