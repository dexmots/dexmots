import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# list files run_dof-i.npz in current directory
for i in range(5):
    if not os.path.exists("run_dof-{}.npz".format(i)):
        break
    data = np.load("run_dof-{}.npz".format(i))
    states = data["states"][:, 0, 3]
    actions = data["ac"].squeeze()[:, 0]
    plt.plot(actions, label="Desired".format(i), alpha=0.5, color="r")
    plt.plot(states, label="Actual".format(i), alpha=0.5, color="b")
    plt.xlabel("Timestep")
    plt.ylabel("Joint state")
    plt.legend()
    plt.show()
