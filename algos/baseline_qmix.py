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
from utils import plot_env_history


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
    total_timesteps: int = int(1e6)
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4 #3e-4 #7e-4 #
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    num_envs: int = 1
    """the number of parallel game environments"""

    # QMIX arguments
    episode_length: int = 2048
    """side of an trajectory to store in buffer size"""

    # DQN arguments
    buffer_size: int = 500
    """the replay memory buffer size in episode"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size (in episodes) of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 500 #10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    pretrained_models: str = None
    """Path to pretrained models"""
    hidden_layer_nn: Union[bool, tuple[int]] = (64,)
    """number of neurons in hidden layer"""
    debug: bool = False
    """debug mode saves monitoring logs during training"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""

    num_agents: int = 1
    """the number of agents in the environment"""
    reward_shaping: str = ""
    """Toggle learning rate annealing for policy and value networks"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
class QMixer(nn.Module):
    def __init__(self, num_agents, observation_space, hidden_dim):
        super().__init__()

        """"
        Excerpt from QMIX paper
            "The mixing network consists of a single hidden layer of 32 units, utilising an ELU non-linearity. 
            
            The hypernetworks are then sized to produce weights of appropriate size. 
            The hypernetwork producing the final bias of the mixing network consists of a single hidden layer 
            of 32 units with a ReLU non-linearity
            
            Each hypernetwork takes the states as input and generates the weights of one layer of the mixing network. 
            Each hypernetwork consists of a single linear layer, followed by an absolute activation function, to
            ensure that the mixing network weights are non-negative.

            The output of the hypernetwork is then a vector, which is reshaped into a matrix of appropriate size. 
            The biases are produced in the same manner but are not restricted to being non-negative. 
            The final bias is produced by a 2 layer hypernetwork with a ReLU non-linearity."
        """
        self.observation_space = observation_space        
        # self.network = nn.Sequential(
        #     layer_init(nn.Linear(num_agents, hidden_dim), std=1.0),
        #     nn.ELU(),
        #     layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        # )
        self.hidden_dim = hidden_dim[0]
        self.num_agents = num_agents

        # Hypernetworks
        self.hyper_network_w1 = layer_init(nn.Linear(np.prod(observation_space.shape), num_agents*self.hidden_dim), std=1.0)
        self.hyper_network_b1 = layer_init(nn.Linear(np.prod(observation_space.shape), self.hidden_dim), std=1.0)
        
        self.hyper_network_w2 = layer_init(nn.Linear(np.prod(observation_space.shape), self.hidden_dim), std=1.0)
        self.hyper_network_b2 = nn.Sequential(
            layer_init(nn.Linear(np.prod(observation_space.shape), self.hidden_dim), std=1.0),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, 1), std=1.0),
        )
        
        self.register_buffer(
            "observation_low", torch.tensor(observation_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "observation_high", torch.tensor(observation_space.high, dtype=torch.float32)
        )

    def forward(self, qvalues, state):
        """
        qvalues (batch_n, num_agent) B, M
        state   (batch_n, dim)
        """
        x = (state - self.observation_low)/(self.observation_high - self.observation_low)
        w1 = torch.abs(self.hyper_network_w1(x))
        w1 = w1.view(-1, self.num_agents, self.hidden_dim)
        w2 = torch.abs(self.hyper_network_w2(x))
        # w2 = w2.view(-1, self.hidden_dim)
        b1 = self.hyper_network_b1(x)
        # b1 = b1.view(1, self.hidden_dim)
        b2 = self.hyper_network_b2(x)
        
        # QMixer network
        out = F.elu(qvalues[:, None,:]@w1 + b1[:, None,:])
        out = out@w2[:, :, None] + b2[:, :, None]
        return out
        
        # self.network[0].weight.data.copy_(w1.reshape(self.num_agents, self.hidden_dim))
        # self.network[0].bias.data.copy_(b1.flatten())
        
        # self.network[2].weight.data.copy_(w2.flatten())
        # self.network[2].bias.data.copy_(b2.flatten())

class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dims):
        super().__init__()
        action_dim = action_space.n
        self.observation_space = observation_space
        self.hidden_dim = hidden_dims[0]
        self.register_buffer(
            "input_low", torch.tensor(np.r_[observation_space.low, np.zeros(action_dim)], dtype=torch.float32)
        )
        self.register_buffer(
            "input_high", torch.tensor(np.r_[observation_space.high, np.ones(action_dim)], dtype=torch.float32)
        )
        self.l1 = layer_init(nn.Linear(np.prod(observation_space.shape) + action_dim, hidden_dims[0]), std=1.0)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.l2 = layer_init(nn.Linear(hidden_dims[-1], action_dim), std=1.0)
        self.reset_hidden_state()

    def reset_hidden_state(self, batch_size=1):
        # https://github.com/oxwhirl/pymarl/blob/master/src/modules/agents/rnn_agent.py
        self.hidden_state = self.l1.weight.new(batch_size, self.hidden_dim).zero_()

    def forward(self, x):
        x = (x - self.input_low)/(self.input_high - self.input_low)
        x = F.relu(self.l1(x)).view(-1, self.hidden_dim)
        h = self.rnn(x, self.hidden_state.view(-1, self.hidden_dim))
        q = self.l2(h)
        self.hidden_state = h
        return q


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    args = tyro.cli(Args)
    controls = ["yaw"]
    env = envs.make(
        args.env_id,
        controls=controls, 
        max_num_steps=args.episode_length,
    )
    args.num_agents = env.num_turbines
    args.reward_shaping = ""
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    # TRY NOT TO MODIFY
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    obs_space = env.observation_space(env.possible_agents[0])
    action_space_extractor = VectorExtractor(env.action_space(env.possible_agents[0]))
    partial_obs_extractor = VectorExtractor(obs_space)
    global_obs_extractor = VectorExtractor(env.state_space)
    partial_obs_space = partial_obs_extractor.space
    global_obs_space = global_obs_extractor.space
    # action_space = action_space_extractor.space
    action_space = gym.spaces.Discrete(3)
    hidden_layer_nn = [] if not isinstance(args.hidden_layer_nn, tuple) else args.hidden_layer_nn
    q_networks = [
        QNetwork(partial_obs_space, action_space, hidden_layer_nn).to(device)
        for _ in range(args.num_agents)
    ]
    target_q_networks = [
        QNetwork(partial_obs_space, action_space, hidden_layer_nn).to(device)
        for _ in range(args.num_agents)
    ]
    qmixer = QMixer(args.num_agents, global_obs_space, hidden_layer_nn).to(device)
    target_qmixer = QMixer(args.num_agents, global_obs_space, hidden_layer_nn).to(device)
   
    optimizer = optim.Adam(
        [param for q_network in q_networks for param in q_network.parameters()] + list(qmixer.parameters()),
        lr=args.learning_rate, eps=1e-5
    )

    # ALGO Logic: Storage setup
    buffer_global_obs = torch.zeros((args.buffer_size, args.episode_length) + global_obs_space.shape).to(device)
    buffer_next_global_obs = torch.zeros((args.buffer_size, args.episode_length) + global_obs_space.shape).to(device)
    buffer_obs = torch.zeros((args.buffer_size, args.episode_length, args.num_agents) + partial_obs_space.shape).to(device)
    buffer_next_obs = torch.zeros((args.buffer_size, args.episode_length, args.num_agents) + partial_obs_space.shape).to(device)
    buffer_actions = torch.zeros((args.buffer_size, args.episode_length, args.num_agents) + (action_space.n,)).to(device)
    buffer_rewards = torch.zeros((args.buffer_size, args.episode_length, args.num_agents)).to(device)
    buffer_dones = torch.zeros((args.buffer_size, args.episode_length, args.num_agents)).to(device)

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
    
    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    env.reset(options={"wind_speed": 8, "wind_direction": 270}) 
    agents_list = env.possible_agents
    assert len(agents_list) == args.num_agents
    cumul_rewards = 0
    cumul_power = 0
    episode_id = 0
    pt = 0

    obs = np.zeros((args.num_agents,) + partial_obs_space.shape)
    next_obs = np.zeros((args.num_agents,) + partial_obs_space.shape)
    actions = np.zeros((args.num_agents,) + (action_space.n,))
    rewards = np.zeros((args.num_agents,))
    dones = np.zeros((args.num_agents,))
    
    for global_step in range(args.total_timesteps):

        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

        # ALGO LOGIC: action logic
        powers = []
        with torch.no_grad():
            last_global_obs = env.state()
            last_global_obs = torch.Tensor(global_obs_extractor(last_global_obs)).to(device)
            for idagent, q_network in enumerate(q_networks):
                last_obs, reward, terminations, truncations, infos = env.last()
                last_obs = partial_obs_extractor(last_obs)
                
                if random.random() < epsilon:
                    action = action_space.sample()
                    action = np.array(action)
                else:
                    input = torch.Tensor(np.concatenate([last_obs, actions[idagent]])).to(device)
                    q_values = q_network(input)
                    action = torch.argmax(q_values, dim=-1).cpu().numpy()
                
                last_done = np.logical_or(terminations, truncations)
                # last_done = torch.Tensor([last_done.astype(int)]).to(device)

                # store values
                # values[step, :, idagent] = value.flatten()
                obs[idagent] = last_obs
                dones[idagent] = last_done
                rewards[idagent] = reward[0]
                one_hot_action = torch.zeros(action_space.n).to(device)
                one_hot_action[action] = 1
                actions[idagent] = one_hot_action

                if "power" in infos:
                    powers.append(infos["power"])
                    writer.add_scalar(f"farm/power_T{idagent}", infos["power"], global_step)
                if "load" in infos:
                    writer.add_scalar(f"farm/load_T{idagent}", sum(np.abs(infos["load"])), global_step)
                writer.add_scalar(f"farm/controls/yaw_T{idagent}", last_obs[0].item(), global_step)

                # store next action
                #TODO: fix continuous_control=False to directly handle discretization
                env.step(action_space_extractor.make_dict(action[None]-1))
            writer.add_scalar(f"farm/reward", float(reward[0]), global_step)

        if "power" in infos:
            writer.add_scalar(f"farm/power_total", sum(powers), global_step)
            cumul_power += sum(powers)
        cumul_rewards += float(reward[0])

    
        # handle `final_observation`
        for idagent, agent_name in enumerate(agents_list):
            next_obs[idagent] = torch.Tensor(partial_obs_extractor(env.observe(agent_name))).to(device)
        next_global_obs =  torch.Tensor(global_obs_extractor(env.state())).to(device)
        # FILL BUFFER
        buffer_id = episode_id % args.buffer_size
        buffer_global_obs[buffer_id, pt, :] = last_global_obs
        buffer_next_global_obs[buffer_id, pt, :] = next_global_obs
        buffer_obs[buffer_id, pt, :] = torch.Tensor(obs).to(device)
        buffer_next_obs[buffer_id, pt, :] = torch.Tensor(next_obs).to(device)
        buffer_actions[buffer_id, pt, :] = torch.Tensor(actions).to(device) 
        buffer_rewards[buffer_id, pt, :] = torch.Tensor(rewards).to(device)
        buffer_dones[buffer_id, pt, :] = torch.Tensor(dones).to(device)
        pt += 1

        # ALGO LOGIC: training.
        if any(dones):
            assert all(dones)
            writer.add_scalar(f"farm/episode_reward", float(cumul_rewards), global_step)
            writer.add_scalar(f"farm/episode_power", float(cumul_power) / args.train_frequency, global_step)
            cumul_rewards = 0
            cumul_power = 0
            env.reset(options={"wind_speed": 8, "wind_direction": 270})
            episode_id += 1
            pt = 0

            if global_step > args.learning_starts:

                # Optimizing the value network
                b_inds = np.random.choice(np.arange(min(episode_id, args.buffer_size)), args.batch_size)
                qval_preds = []
                targets = []
                for idagent, (q_network, target_network) in enumerate(zip(q_networks, target_q_networks)):
                    trajectory_targets = []
                    trajectory_qval_preds = []
                    target_network.reset_hidden_state(args.batch_size)
                    q_network.reset_hidden_state(args.batch_size)
                    
                    for t in range(1, args.episode_length):
                        with torch.no_grad():
                            input = torch.cat([buffer_next_obs[b_inds, t, idagent], buffer_actions[b_inds, t, idagent]], 1)
                            target_max, _ = target_q_networks[idagent](input).max(dim=1)
                            # v_next_max = args.gamma * target_max * (1 - buffer_dones[b_inds, t, idagent].flatten())
                            # td_target = buffer_rewards[b_inds, t, idagent].flatten() + v_next_max
                            trajectory_targets.append(target_max)

                        input = torch.cat([buffer_obs[b_inds, t, idagent], buffer_actions[b_inds, t-1, idagent]], 1)
                        action_indices = torch.where(buffer_actions[b_inds, t, idagent])[1]
                        old_val = q_network(input).gather(1, action_indices[None]).squeeze()
                        trajectory_qval_preds.append(old_val)
                    
                    targets.append(torch.stack(trajectory_targets, dim=1))
                    qval_preds.append(torch.stack(trajectory_qval_preds, dim=1))
                
                qval_preds = torch.stack(qval_preds, dim=-1).view(-1,action_space.n)
                targets = torch.stack(targets, dim=-1).view(-1,action_space.n)
                
                with torch.no_grad():
                    b_global_obs = buffer_next_global_obs[b_inds, 1:, :].view((-1,)+global_obs_space.shape)
                    target_max_qmix = target_qmixer(targets, b_global_obs)
                    v_next_max_qmix = args.gamma * target_max_qmix.flatten() * (1 - buffer_dones[b_inds, 1:, 0].flatten())
                    td_target = buffer_rewards[b_inds, 1:, 0].flatten() + v_next_max_qmix
                qmixer_old_val = qmixer(qval_preds, buffer_global_obs[b_inds, 1:, :].view((-1,)+global_obs_space.shape))
                
                loss = F.mse_loss(td_target, qmixer_old_val.squeeze())

                if global_step % 100 == 0:
                    writer.add_scalar(f"losses/agent_{idagent}/td_loss", loss, global_step)
                    writer.add_scalar(f"losses/agent_{idagent}/q_values", qmixer_old_val.mean().item(), global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                    # update target network
            if global_step % args.target_network_frequency == 0:
                for q_network, target_network in zip(q_networks, target_q_networks):
                    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )
                for target_network_param, q_network_param in zip(target_qmixer.parameters(), qmixer.parameters()):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )
        
            for q_network in q_networks:
                q_network.reset_hidden_state()
            for target_network in target_q_networks:
                target_network.reset_hidden_state()
        
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