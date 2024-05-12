import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns


sns.set_theme(style="darkgrid")


def plot_env_history(env):
    columns = [f"T{i+1}" for i in range(env.num_turbines)]
    yaws = np.c_[[[h["yaw"] for h in env.history[agent]["observation"]]for agent in env.possible_agents]].T
    powers = np.c_[[env.history[agent]["power"] for agent in env.possible_agents]].T
    yaws = pd.DataFrame(yaws, columns=columns)
    powers = pd.DataFrame(powers, columns=columns)
    fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
    ax0 = sns.lineplot(yaws, ax=ax[0])
    ax1 = sns.lineplot(powers.sum(1), ax=ax[1])
    ax0.set(ylabel="Yaw (°)", xlabel="Iterations")
    ax1.set(ylabel="Power (MW)", xlabel="Iterations")
    plt.grid()
    return fig