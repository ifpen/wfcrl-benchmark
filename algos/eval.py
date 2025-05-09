import os
import random
import time
from dataclasses import dataclass
import pickle
from typing import Union

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import tyro


from wfcrl import environments as envs

from extractors import VectorExtractor
from utils import (
    eval_wind_rose,
    eval_policies,
    get_env_history,
    multi_agent_step_routine,
    multidiscrete_one_hot,
    plot_env_history,
    prepare_eval_windrose,
)

def recurrent_q_net_deterministic_act(q_network, observation, last_action):
    if last_action is None:
        last_action = np.zeros(sum(action_space.nvec))
    else:
        last_action = multidiscrete_one_hot(action_space_extractor(last_action), action_space)
    observation = partial_obs_extractor(observation)
    input = torch.tensor(np.concatenate([observation, last_action]), dtype=torch.float32, device=device)
    q_values = q_network(input)
    action = torch.argmax(q_values, dim=-1).cpu().numpy()
    return action_space_extractor.make_dict(np.array([action]))

def ALGO_TO_AGENTS(algo):
    match algo:
        case "ippo":
            from baseline_ippo import Agent
            def get_deterministic_action(agent, observation, last_action=None):
                observation = torch.Tensor(partial_obs_extractor(observation)).to(device)
                action, _, _, _= agent.get_action_and_value(observation, deterministic=True)
                return action_space_extractor.make_dict(action)
        case "mappo":
            from baseline_mappo import Agent
            def get_deterministic_action(agent, observation, last_action=None):
                observation = torch.Tensor(partial_obs_extractor(observation)).to(device)
                action, _, _ = agent.get_action(observation, deterministic=True)
                return action_space_extractor.make_dict(action)
        case "idqn":
            from baseline_idqn import QNetwork as Agent
            def get_deterministic_action(q_network, observation, last_action):
                observation = torch.tensor(partial_obs_extractor(observation), dtype=torch.float32, device=device)
                q_values = q_network(observation)
                action = torch.argmax(q_values, dim=-1).cpu().numpy()
                return action_space_extractor.make_dict(np.array([action]))
        case "idrqn":
            from baseline_idrqn import QNetwork as Agent
            get_deterministic_action = recurrent_q_net_deterministic_act 
        case "qmix":
            from baseline_qmix import QNetwork as Agent
            get_deterministic_action = recurrent_q_net_deterministic_act 
        case _:
            print(f"Unknown algorithm {algo}")
    return Agent, get_deterministic_action

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    scenario: str = "windrose"
    """the name of the scenario, identified by the wind series"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    load_coef: float = 1
    """coefficient of the load penalty"""
    episode_length: int = 150
    """side of an trajectory to store in buffer size"""
    wind_data: str = "data/smarteole.csv"
    """Path to wind data for wind rose evaluation"""
    env_id: str = "Dec_Turb3_Row1_Floris"
    """the id of the environment"""
    num_episodes: int = 1
    """the number of iterations (computed in runtime)"""
    algo: str = "ippo"
    """ the name of the trained algorithm"""

    # model arguments
    pretrained_models: str = "path/to/model" # 
    """Path to pretrained models"""
    output_folder: str = "eval"
    """output folder"""
    policy: str = "base"
    """Type of policy"""
    hidden_layer_nn: Union[bool, tuple[int]] = (64, 64)
    """number of neurons in hidden layer"""
    debug: bool = False
    """debug mode saves monitoring logs during training"""
    num_agents: int = 1
    """the number of agents in the environment"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    controls = ["yaw"]
    algo = args.algo
    assert algo in ["ippo", "mappo", "idqn", "idrqn", "qmix"]
    env = envs.make(
        args.env_id,
        controls=controls, 
        max_num_steps=args.episode_length,
        load_coef=args.load_coef,
        continuous_control=algo in ["mappo", "ippo"],
    )
    args.num_agents = env.num_turbines
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    Path(f"{args.pretrained_models}/{args.output_folder}").mkdir(exist_ok=True, parents=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    obs_space = env.observation_space(env.possible_agents[0])
    action_space_extractor = VectorExtractor(env.action_space(env.possible_agents[0]))
    partial_obs_extractor = VectorExtractor(obs_space, filter_out=["pitch", "torque"])
    partial_obs_space = partial_obs_extractor.space
    action_space = action_space_extractor.space
    hidden_layer_nn = [] if not isinstance(args.hidden_layer_nn, tuple) else args.hidden_layer_nn
    Agent, get_deterministic_action = ALGO_TO_AGENTS(algo)
    agents = [
        Agent(partial_obs_space, action_space, hidden_layer_nn).eval().to(device)
        for _ in range(args.num_agents)
    ]

    args.pretrained_models = Path(args.pretrained_models)
    assert args.pretrained_models.exists()
    for idagent, agent in enumerate(agents):
        try:
            path = list(args.pretrained_models.glob(f"*model_{idagent}"))[0]
        except:
            raise FileNotFoundError(f"No file in model_{idagent} found under folder {args.pretrained_models}")
        params = torch.load(str(path), map_location='cpu')
        agent.load_state_dict(params)

    if args.scenario == "windrose":
        windrose = prepare_eval_windrose(args.wind_data, num_bins=5)
        def evaluate(eval_env, eval_agents):
            return eval_wind_rose(eval_env, eval_agents, windrose, get_deterministic_action)[0]
        env.reset(args.seed)
    else:
        def evaluate(eval_env, eval_agents):
            return eval_policies(eval_env, eval_agents, get_deterministic_action)
        env.reset(options={"wind_speed": 8, "wind_direction": 270})

    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    #TODO: put seed back 
    # env.reset(seed=args.seed)

    episode_rewards = []
    episode_loads = []
    episode_powers = []
    episode_yaws = []

    for iteration in range(1, args.num_episodes + 1):
        if args.scenario == "windrose":
            score, windrose_r, bins = eval_wind_rose(env, agents, windrose, get_deterministic_action)
            pd.DataFrame(np.c_[bins, windrose_r]).to_csv(args.pretrained_models/f"{args.output_folder}/windrose_scores.csv", index=False)
        else:
            score = eval_policies(env, agents, get_deterministic_action)
            yaws, powers, loads, rewards = get_env_history(env)
            # all policies have received the same reward
            episode_rewards.append(rewards.squeeze())
            episode_loads.append(loads)
            episode_powers.append(powers)
            episode_yaws.append(yaws)

            pd.DataFrame(yaws).to_csv(args.pretrained_models/f"{args.output_folder}/yaws.csv", index=False)
            pd.DataFrame(loads).to_csv(args.pretrained_models/f"{args.output_folder}/loads.csv", index=False)
            pd.DataFrame(powers).to_csv(args.pretrained_models/f"{args.output_folder}/powers.csv", index=False)
            pd.DataFrame(rewards.squeeze()).to_csv(args.pretrained_models/f"{args.output_folder}/rewards.csv", index=False)
            # with open(f"{args.output_folder}/{run_name}/history_{iteration}.pickle", 'wb') as f:
            #     pickle.dump(env.history, f, pickle.HIGHEST_PROTOCOL)
            try:
                fig = plot_env_history(env)
                fig.savefig(f"{args.pretrained_models}/{args.output_folder}/plot_iter{iteration}.png")
            except:
                print("Could not save figure.")
    
    env.close()
    # writer.close()