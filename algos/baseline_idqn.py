import os
import random
import time
from dataclasses import dataclass
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
from utils import LocalSummaryWriter, plot_env_history


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
    track: bool = True
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
    total_timesteps: int = int(1e5)
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4 #3e-4 #7e-4 #
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_envs: int = 1
    """the number of parallel game environments"""

    # DQN arguments
    buffer_size: int = 32 #128
    """the replay memory buffer size"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 0.3
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 1000 #10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    pretrained_models: str = None
    """Path to pretrained models"""
    hidden_layer_nn: Union[bool, tuple[int]] = (64,) #(120, 84)
    """number of neurons in hidden layer"""
    debug: bool = False
    """debug mode saves monitoring logs during training"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""

    # to be filled in runtime
    num_agents: int = 1
    """the number of agents in the environment"""
    reward_shaping: str = ""
    """Toggle learning rate annealing for policy and value networks"""
    device: str = "cpu"
    "device"

class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_layers=(120,84)):
        super().__init__()
        self.observation_space = observation_space
        input_layers = [np.prod(observation_space.shape)] + list(hidden_layers)
        self.register_buffer(
            "observation_low", torch.tensor(observation_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "observation_high", torch.tensor(observation_space.high, dtype=torch.float32)
        )

        self.network = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
                for in_dim, out_dim in zip(input_layers[:-1], input_layers[1:])
            ],
        )
        self.output_layers = nn.ModuleList([
            nn.Linear(input_layers[-1], space.n)
            for space in action_space
        ])
        
    def forward(self, x):
        x = (x - self.observation_low)/(self.observation_high - self.observation_low)
        x = self.network(x)
        x = torch.cat([net(x) for net in self.output_layers])
        return x


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    args = tyro.cli(Args)
    controls = {"yaw": (-40, 40)}
    env = envs.make(
        args.env_id,
        controls=controls, 
        max_num_steps=2048,
        continuous_control=False # discrete action space for DQN
    )
    args.num_agents = env.num_turbines
    args.reward_shaping = ""
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device
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
    writer = LocalSummaryWriter(f"runs/{run_name}")
    writer.add_config(vars(args))
    model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    # TRY NOT TO MODIFY
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    obs_space = env.observation_space(env.possible_agents[0])
    action_space_extractor = VectorExtractor(env.action_space(env.possible_agents[0]))
    partial_obs_extractor = VectorExtractor(obs_space)
    partial_obs_space = partial_obs_extractor.space
    action_space = action_space_extractor.space
    
    hidden_layer_nn = args.hidden_layer_nn if isinstance(args.hidden_layer_nn, tuple) else []
    q_networks = [
        QNetwork(partial_obs_space, action_space, hidden_layer_nn).to(device)
        for _ in range(args.num_agents)
    ]
    target_q_networks = [
        QNetwork(partial_obs_space, action_space, hidden_layer_nn).to(device)
        for _ in range(args.num_agents)
    ]
    optimizers = [
        optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=1e-5)
        for q_network in q_networks
    ]

    if args.pretrained_models is not None:
        args.pretrained_models = Path(args.pretrained_models)
        assert args.pretrained_models.exists()
        for idnetwork, q_network in enumerate(q_networks):
            try:
                path = list(args.pretrained_models.glob(f"*model_{idnetwork}"))[0]
            except:
                raise FileNotFoundError(f"No file in model_{idnetwork} found under folder {args.pretrained_models}")
            params = torch.load(str(path), map_location='cpu')
            q_network.load_state_dict(params)
    
    for q_network, target_q_network in zip(q_networks, target_q_networks):
        target_q_network.load_state_dict(q_network.state_dict())
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.buffer_size, args.num_agents) + partial_obs_space.shape, device=device)
    next_obs = torch.zeros((args.buffer_size, args.num_agents) + partial_obs_space.shape, device=device)
    actions = torch.zeros((args.buffer_size, args.num_agents) + action_space.shape, dtype=torch.int64, device=device)
    rewards = torch.zeros((args.buffer_size, args.num_agents), device=device)
    dones = torch.zeros((args.buffer_size, args.num_agents), dtype=torch.int64, device=device)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    env.reset(options={"wind_speed": 8, "wind_direction": 270}) 
    agents_list = env.possible_agents
    assert len(agents_list) == args.num_agents
    cumul_rewards = 0
    cumul_power = 0
    last_done = False

    for global_step in range(args.total_timesteps):
        # progressively replace old data to handle non stationarity
        if last_done:
            writer.add_scalar(f"farm/episode_reward", float(cumul_rewards), global_step)
            writer.add_scalar(f"farm/episode_power", float(cumul_power) / env.max_num_steps, global_step)
            cumul_rewards = 0
            cumul_power = 0
            dones[buffer_id, :] = torch.tensor(1, device=device)
            env.reset(options={"wind_speed": 8, "wind_direction": 270})

        buffer_id = global_step % args.buffer_size

        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        # ALGO LOGIC: action logic
        powers = []
        with torch.no_grad():
            for idagent, q_network in enumerate(q_networks):
                last_obs, reward, terminations, truncations, infos = env.last()
                last_obs = partial_obs_extractor(last_obs)
                if random.random() < epsilon:
                    action = action_space.sample()
                else:
                    q_values = q_network(torch.tensor(last_obs, device=device))
                    action = torch.argmax(q_values, dim=-1).cpu().numpy()[None]
                
                last_done = np.logical_or(terminations, truncations)

                # store values
                obs[buffer_id, idagent] = torch.tensor(last_obs, device=device)
                dones[buffer_id, idagent] = torch.tensor(int(last_done), device=device)
                rewards[buffer_id, idagent] = torch.tensor(reward[0], device=device)
                actions[buffer_id, idagent] = torch.tensor(action, dtype=torch.int64, device=device)

                if "power" in infos:
                    powers.append(infos["power"])
                    writer.add_scalar(f"farm/power_T{idagent}", infos["power"], global_step)
                if "load" in infos:
                    writer.add_scalar(f"farm/load_T{idagent}", sum(np.abs(infos["load"])), global_step)
                writer.add_scalar(f"farm/controls/yaw_T{idagent}", last_obs[0].item(), global_step)
                writer.add_scalar(f"charts/action/yaw_T{idagent}", action[0], global_step)

                # store next action
                env.step(action_space_extractor.make_dict(action))
            writer.add_scalar(f"farm/reward", float(reward[0]), global_step)
        
        writer.add_scalar(f"charts/epsilon", epsilon, global_step)
        if "power" in infos:
            writer.add_scalar(f"farm/power_total", sum(powers), global_step)
            cumul_power += sum(powers)
        cumul_rewards += float(reward[0])
    
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        for idagent, agent_name in enumerate(agents_list):
            next_obs[buffer_id, idagent] = torch.tensor(partial_obs_extractor(env.observe(agent_name)), device=device)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        if global_step > args.buffer_size:
            if global_step % args.train_frequency == 0:

                # data = rb.sample(args.batch_size)

        # Optimizing the value network
                for idagent, q_network in enumerate(q_networks):
                    b_inds = np.arange(args.batch_size)
                    # np.random.shuffle(b_inds)

                    with torch.no_grad():
                        target_max, _ = target_q_networks[idagent](next_obs[b_inds, idagent]).max(dim=1)
                        td_target = rewards[b_inds, idagent].flatten() + args.gamma * target_max * (1 - dones[b_inds, idagent].flatten())
                    old_val = q_network(obs[b_inds, idagent]).gather(1, actions[b_inds, idagent]).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 100 == 0:
                        writer.add_scalar(f"losses/agent_{idagent}/td_loss", loss.item(), global_step)
                        writer.add_scalar(f"losses/agent_{idagent}/q_values", old_val.mean().item(), global_step)
                        # print("SPS:", int(global_step / (time.time() - start_time)))

                    # optimize the model
                    optimizers[idagent].zero_grad()
                    loss.backward()
                    optimizers[idagent].step()

                    # update target network
            if global_step % args.target_network_frequency == 0:
                for q_network, target_network in zip(q_networks, target_q_networks):
                    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )

        if (global_step % 10000 == 0) and args.save_model:
            for idagent, q_network in enumerate(q_networks):
                torch.save(q_network.state_dict(), model_path+f"_{idagent}")
            # torch.save(shared_critic.state_dict(), model_path+f"_critic")
            print(f"model saved to {model_path}")
        
            print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    
env.close()
for idagent, q_network in enumerate(q_networks):
    torch.save(q_network.state_dict(), model_path+f"_{idagent}")
# torch.save(shared_critic.state_dict(), model_path+f"_critic")
print(f"model saved to {model_path}")
writer.close()


# Prepare plots
fig = plot_env_history(env)
fig.savefig(f"runs/{run_name}/plot.png")

print("stop")