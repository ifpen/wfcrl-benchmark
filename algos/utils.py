import decimal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="darkgrid")

def get_env_history(env):
    columns = [f"T{i+1}" for i in range(env.num_turbines)]
    yaws = np.c_[[[h["yaw"] for h in env.history[agent]["observation"]]for agent in env.possible_agents]].T
    powers = np.c_[[env.history[agent]["power"] for agent in env.possible_agents]].T
    loads = np.c_[[env.history[agent]["load"] for agent in env.possible_agents]].T
    rewards = np.c_[[env.history[agent]["reward"] for agent in env.possible_agents]].T
    yaws = pd.DataFrame(yaws, columns=columns)
    powers = pd.DataFrame(powers, columns=columns)
    loads = pd.DataFrame(np.abs(loads).sum(0), columns=columns)
    return yaws, powers, loads, rewards

def plot_env_history(env):
    yaws, powers, loads, _ = get_env_history(env)
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    ax0 = sns.lineplot(yaws, ax=ax[0])
    ax1 = sns.lineplot(powers.sum(1), ax=ax[1])
    ax2 = sns.lineplot(loads.sum(1), ax=ax[2])
    ax0.set(ylabel="Yaw (°)", xlabel="Iterations")
    ax1.set(ylabel="Normalized Power (MW)", xlabel="Iterations")
    ax2.set(ylabel="Loading Indicator", xlabel="Iterations")
    ax0.grid(True)
    ax1.grid(True)
    ax2.grid(True)
    return fig

def less_than_180(angle):
    # modulo for [-180, 180]
    # takes an angle **already** in the [-180, 180] referential
    return -180 + (angle - (-180)) % 360

def get_wake_delays(cx, cy, uref, phiref=0, gamma=4, stab_time = 80, cutoff_x=2000, cutoff_y=400):
    # Store number of decimals
    precision = -decimal.Decimal(cx[0]).as_tuple().exponent
    # Rotate to get downstream coordinates
    # Remove 0 coords to calculate angle 
    cx = np.array(cx)
    cx[cx==0] = 1
    phiref = less_than_180(phiref)

    # Get polar coordinates
    rs = np.sqrt(cx**2 + np.array(cy)**2)
    theta = np.arctan2(cy/rs, cx/rs)
    theta_new = theta + np.radians(phiref)
    cx = np.round(rs * np.cos(theta_new), precision)
    cy = np.round(rs * np.sin(theta_new), precision)

    nturb = len(cx)
    D = np.ones((nturb,nturb)) * stab_time
    for i in range(nturb):
        for j in range(nturb):
            if (cx[j] > cx[i]) and (cx[j] < cx[i] + cutoff_x) and np.abs(cy[j] - cy[i]) < cutoff_y:
                D[i,j] = max(stab_time, gamma * (cx[j] - cx[i]) / uref)
    return D

def multi_agent_step_routine(env, policies, get_action=None):
    r = {agent: 0 for agent in env.possible_agents}
    if get_action is None:
        def get_action(agt, obs):
            return agt(obs)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        r[agent] += reward
        action = get_action(policies[env.agent_name_mapping[agent]], observation)
        env.step(action)
    return r

def prepare_eval_windrose(path, num_bins=5):
    """
    to plot the 2D histogram:
    ax = sns.heatmap(freq, annot=True, cbar=False)
    ax.set_xticks(np.arange(6), labels=np.round(ws_bins).astype(int))
    ax.set_yticks(np.arange(6), labels=np.round(wd_bins).astype(int))
    ax.set(xlabel="speed", ylabel="direction")
    """
    df = pd.read_csv(path)
    wd = df["wd"].values % 360
    wd = (wd + 60) % 360
    ws = df["ws"].values
    counts, wd_bins, ws_bins =  np.histogram2d(wd, ws, bins=5)
    freq = counts / np.sum(counts)
    return freq, wd_bins, ws_bins
    

def eval_wind_rose(env, policies, wind_rose, get_action=None):
    """
    Obtained from SMARTEOLE data centered on turbine row

    train = pd.read_csv(path_to_smarteole_data)
    train["wd"] = train["wd"] % 360
    wd = (train["wd"] + 60) % 360
    wd_nums, wd_bins, _ = plt.hist(wd.values, bins=10)
    ws_nums, ws_bins, _ = plt.hist(train["ws"].values, bins=10)
    """
    
    freq, wd_bins, ws_bins = wind_rose
    freq = np.atleast_2d(freq)
    wd_values = (wd_bins[:-1] + wd_bins[1:])/2
    ws_values = (ws_bins[:-1] + ws_bins[1:])/2
    num_episodes = freq.size
    print(f"Evaluating on {num_episodes} episodes")
    episode_rewards = []
    score = 0

    for i, wd in enumerate(wd_values):
        for j, ws in enumerate(ws_values):
            env.reset(options={"wind_speed": ws, "wind_direction": wd})
            r = multi_agent_step_routine(env, policies, get_action=get_action)
            # all policies have received the same reward
            r = float(r[env.possible_agents[0]])
            episode_rewards.append(r)
            score += freq[i, j] * r
    return score, np.array(episode_rewards)