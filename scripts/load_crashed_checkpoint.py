import torch
import yaml

from shac.algorithms.shac import SHAC
from shac.algorithms.shac import SHAC
from shac.utils import torch_utils as tu
from dmanip.utils.common import *


def draw_graph(start, watch=[], filename="net_graph.jpg"):
    from graphviz import Digraph
    import pylab

    node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="12",
        ranksep="0.1",
        height="0.2",
    )
    graph = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    assert hasattr(start, "grad_fn")
    if start.grad_fn is not None:
        _draw_graph(start.grad_fn, graph, watch=watch)

    size_per_element = 0.15
    min_size = 12

    # Get the approximate number of nodes and edges
    num_rows = len(graph.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    graph.graph_attr.update(size=size_str)

    graph.render(filename=filename)
    filename = graph.render(filename="img/g1")
    pylab.savefig(filename)


def _draw_graph(var, graph, watch=[], seen=[], indent="", pobj=None):
    """recursive function going through the hierarchical graph printing off
    what we need to see what autograd is doing."""
    from rich import print

    if hasattr(var, "next_functions"):
        for fun in var.next_functions:
            joy = fun[0]
            if joy is not None:
                if joy not in seen:
                    label = (
                        str(type(joy))
                        .replace("class", "")
                        .replace("'", "")
                        .replace(" ", "")
                    )
                    label_graph = label
                    colour_graph = ""
                    seen.append(joy)

                    if hasattr(joy, "variable"):
                        happy = joy.variable
                        if happy.is_leaf:
                            label += " \U0001F343"
                            colour_graph = "green"

                            for (name, obj) in watch:
                                if obj is happy:
                                    label += (
                                        " \U000023E9 "
                                        + "[b][u][color=#FF00FF]"
                                        + name
                                        + "[/color][/u][/b]"
                                    )
                                    label_graph += name

                                    colour_graph = "blue"
                                    break

                            if len(watch) > 0:
                                vv = [str(obj.shape[x]) for x in range(len(obj.shape))]
                                label += " [["
                                label += ", ".join(vv)
                                label += "]]"
                                label += " " + str(happy.var())

                    graph.node(str(joy), label_graph, fillcolor=colour_graph)
                    print(indent + label)
                    _draw_graph(joy, graph, watch, seen, indent + ".", joy)
                    if pobj is not None:
                        graph.edge(str(pobj), str(joy))


cfg_train = yaml.load(
    open("logs/ClawWarp/shac-rotate-testrun/cfg.yaml", "r"), Loader=yaml.Loader
)
cfg_train["params"]["diff_env"]["name"] = "ClawWarpEnv"
cfg_train["params"]["config"]["steps_num"] = 16
cfg_train["params"]["config"]["num_actors"] = 32
traj_optimizer = SHAC(cfg_train)
traj_optimizer.load("logs/ClawWarp/shac-rotate-testrun/crashed-training.pt")
cp = torch.load("logs/ClawWarp/shac-rotate-testrun/bad_state.pt")
# cp = {k: v.view(32, -1)[:1] for k, v in cp.items()}
env = traj_optimizer.env
# env.sim_substeps = 1
# env.sim_dt = 1 / 240

env = traj_optimizer.env
for i in range(20):
    env.reset()
    env.load_checkpoint(cp)
    actor_loss = traj_optimizer.compute_actor_loss()
    actor_loss.backward()
    # if i % 10 == 0:
    #     draw_graph(
    #         actor_loss,
    #         watch=[p for p in traj_optimizer.actor.named_parameters()],
    #         filename="actor_loss_graph_{}".format(i),
    #     )
    grad_norm = tu.grad_norm(traj_optimizer.actor.parameters())
    print("grad_norm", grad_norm, "actor loss", actor_loss)
    for p in traj_optimizer.actor.parameters():
        if p.grad is not None:
            p.grad.zero_()
    for p in traj_optimizer.critic.parameters():
        if p.grad is not None:
            p.grad.zero_()
