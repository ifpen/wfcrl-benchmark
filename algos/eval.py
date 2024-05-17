import os
import random
import time
from dataclasses import dataclass
import pickle
from typing import Union

from dotenv import load_dotenv
import gymnasium as gym
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Normal

from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from wfcrl.rewards import StepPercentage, RewardShaper
from wfcrl import environments as envs

from extractors import VectorExtractor
from utils import get_env_history, multi_agent_step_routine, plot_env_history


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Benchmark-WFCRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "Dec_Turb3_Row1_Floris"
    """the id of the environment"""
    total_timesteps: int = int(1e3)
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4 #7e-4 #
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95 #0
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0 #0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    num_envs: int = 1
    """the number of parallel game environments"""

    # DFAC arguments
    pretrained_models: str = "runs/name_of_run"
    """Path to pretrained models"""
    kl_coef:  float = 0.0
    """Weighing coefficient for KL term in loss """ 
    policy: str = "base"
    """Type of policy"""
    adaptive_explo: bool = False
    """Use counter_based exploration"""
    multi_scale: bool = False
    """Use multi scale algorithm"""
    hidden_layer_nn: Union[bool, tuple[int]] = (64, 64)
    """number of neurons in hidden layer"""
    debug: bool = False
    """debug mode saves monitoring logs during training"""
    num_steps: int = 500 #1
    """number of available rewards before update"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    num_agents: int = 1
    """the number of agents in the environment"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    controls = ["yaw"]
    args.num_iterations = args.total_timesteps // args.num_steps
    env = envs.make(
        args.env_id,
        controls=controls, 
        max_num_steps=args.num_steps,
    )
    args.num_agents = env.num_turbines
    # reward_shaper.name
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        # os.environ["HTTPS_PROXY"] = "http://irsrvpxw1-std:8082"
        import wandb
        load_dotenv()
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.tensorboard.patch(root_logdir=f"runs/{run_name}", pytorch=False, tensorboard_x=False, save=False)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            # sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    Path(f"runs/{run_name}").mkdir(exist_ok=True, parents=True)
    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )
    # model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    # TRY NOT TO MODIFY
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
    if args.pretrained_models.find("mappo") > -1:
        from baseline_mappo_windrose import Agent
        def get_deterministic_action(agent, observation):
            observation = torch.Tensor(partial_obs_extractor(observation)).to(device)
            action, _, _ = agent.get_action(observation, deterministic=True)
            return action_space_extractor.make_dict(action)  
    else:
        from baseline_ippo_windrose import Agent
        def get_deterministic_action(agent, observation):
            observation = torch.Tensor(partial_obs_extractor(observation)).to(device)
            action, _, _, _= agent.get_action_and_value(observation, deterministic=True)
            return action_space_extractor.make_dict(action)
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

    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    #TODO: put seed back 
    # env.reset(seed=args.seed)

    episode_rewards = []
    episode_loads = []
    episode_powers = []
    episode_yaws = []

    for iteration in range(1, args.num_iterations + 1):
        env.reset(options={"wind_speed": 8, "wind_direction": 270})
        r = multi_agent_step_routine(env, agents, get_action=get_deterministic_action)
        yaws, powers, loads, rewards = get_env_history(env)
        # all policies have received the same reward
        episode_rewards.append(rewards)    
        episode_loads.append(loads)
        episode_powers.append(powers)
        episode_yaws.append(yaws)
        yaws, powers, loads

        # Prepare plots
        with open(f"runs/{run_name}/history_{iteration}.pickle", 'wb') as f:
            pickle.dump(env.history, f, pickle.HIGHEST_PROTOCOL)
        try:
            fig = plot_env_history(env)
            fig.savefig(f"runs/{run_name}/plot_iter{iteration}.png")
        except:
            print("Could not save figure.")
    
    env.close()
    # writer.close()