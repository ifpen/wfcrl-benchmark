import os
import random
import time
from dataclasses import dataclass
from typing import Union

import gymnasium as gym
import numpy as np
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

from extractors import FourierExtractor, DfacSPaceExtractor
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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "Dec_Turb3_Row1_Floris"
    """the id of the environment"""
    total_timesteps: int = int(5e3)
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4 #7e-4 #
    """the learning rate of the optimizer"""
    gamma: float = 0.75
    """the discount factor gamma"""
    gae_lambda: float = 0.95 #0
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01 #0
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
    pretrained_models: str = None
    """Path to pretrained models"""
    reward_tol: float = 0.00005
    """Tolerance threshold for reward function"""
    action_bound: float = 1
    """Bounds on the action space"""
    kl_coef:  float = 0.0
    """Weighing coefficient for KL term in loss """ 
    policy: str = "base"
    """Type of policy"""
    adaptive_explo: bool = False
    """Use counter_based exploration"""
    multi_scale: bool = False
    """Use multi scale algorithm"""
    hidden_layer_nn: Union[bool, tuple[int]] = False #(81,)
    """number of neurons in hidden layer"""
    yaw_max: int = 40
    """maximum absolute yawing in state space""" 
    fourier_order: int = 8
    """order of Fourier basis"""
    fourier_maxdim: int = 81
    """maximum dimension of Fourier basis"""
    fourier_hyper:  bool = False
    """use hypernet arch to learn Fourier basis"""
    fourier_learnable: bool = False
    """learnable Fourier basis"""
    fourier_hyper_arch: tuple = (81,)
    """Architecture of Fourier Hyper"""
    debug: bool = False
    """debug mode saves monitoring logs during training"""
    num_steps: int = 128 #1
    """number of available rewards before update"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    num_agents: int = 1
    """the number of agents in the environment"""
    
def make_yaw_action(action):
    return {"yaw": action.cpu().numpy()}

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class FilteredStep(StepPercentage):
    def __init__(self, reference: float = 0.0, threshold: float = 0.0):
        super().__init__(reference)
        self.threshold = threshold
    
    def __call__(self, reward):
        shaped_reward = 0.0
        if self.reference > 0:
            percentage = (reward - self.reference) / self.reference
            if np.abs(percentage) > self.threshold:
                shaped_reward = np.sign(percentage)
        self.reference = reward
        return shaped_reward

class Agent(nn.Module):
    def __init__(self, observation_space, action_space, hidden_layers, features_extractor_params = {}):
        super().__init__()
        action_dim = action_space.shape[0]

        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        features_extractor = FourierExtractor(observation_space, **features_extractor_params)

        input_layers = [features_extractor.features_dim] + list(hidden_layers)
        self.critic = nn.Sequential(
            features_extractor,
            *[
                nn.Sequential(layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh())
                for in_dim, out_dim in zip(input_layers[:-1], hidden_layers)
            ],
            layer_init(nn.Linear(input_layers[-1], 1), std=1.0),
        )
        self.actor = nn.Sequential(
            features_extractor,
            *[
                nn.Sequential(layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh())
                for in_dim, out_dim in zip(input_layers[:-1], hidden_layers)
            ],
            layer_init(nn.Linear(input_layers[-1], action_dim), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        action_mean = self.actor(x)
        action_std = torch.ones_like(action_mean) * self.log_std.exp()
        distribution = Normal(action_mean, action_std)
        if action is None:
            action = distribution.mode() if deterministic else distribution.rsample()
        return action, distribution.log_prob(action).sum(-1), distribution.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    controls = {"yaw": (-args.yaw_max, args.yaw_max, args.action_bound)}
    env = envs.make(
        args.env_id,
        controls=controls, 
        max_num_steps=args.total_timesteps, 
        reward_shaper=FilteredStep(threshold=args.reward_tol)
    )
    args.num_agents = env.num_turbines
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        os.environ["HTTPS_PROXY"] = "http://irsrvpxw1-std:8082"
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
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
    # TRY NOT TO MODIFY
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    local_obs_space = env.observation_space(env.possible_agents[0])
    global_obs_space = env.state_space
    action_space = env.action_space(env.possible_agents[0])["yaw"]
    partial_obs_extractor = DfacSPaceExtractor(local_obs_space, global_obs_space)
    partial_obs_space = partial_obs_extractor.observation_space
    features_extractor_params = {
        "order":args.fourier_order,
        "hyper": args.fourier_hyper,
        "learnable": args.fourier_learnable,
        "max_dim": args.fourier_maxdim,
        "seed": args.seed,
    }
    hidden_layer_nn = [] if not isinstance(args.hidden_layer_nn, tuple) else args.hidden_layer_nn
    agents = [
        Agent(partial_obs_space, action_space, hidden_layer_nn, features_extractor_params).to(device)
        for _ in range(args.num_agents)
    ]
    optimizers = [
        optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        for agent in agents
    ]

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps+1, args.num_envs, args.num_agents) + partial_obs_space.shape).to(device)
    actions = torch.zeros((args.num_steps+1, args.num_envs, args.num_agents) + action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps+1, args.num_envs, args.num_agents)).to(device)
    rewards = torch.zeros((args.num_steps+1, args.num_envs, args.num_agents)).to(device)
    dones = torch.zeros((args.num_steps+1, args.num_envs, args.num_agents)).to(device)
    values = torch.zeros((args.num_steps+1, args.num_envs, args.num_agents)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    #TODO: put seed back 
    # env.reset(seed=args.seed)
    env.reset(options={"wind_speed": 8, "wind_direction": 270})

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            for optimizer in optimizers:
                optimizer.param_groups[0]["lr"] = lrnow

        values[0] = values[-1]
        logprobs[0] = logprobs[-1]
        obs[0] = obs[-1]
        dones[0] = dones[-1]
        rewards[0] = rewards[-1]
        actions[0] = actions[-1]

        for step in range(1, args.num_steps+1):

            global_step += args.num_envs
            # obs = next_obs

            # ALGO LOGIC: action logic
            powers = []
            with torch.no_grad():
                for idagent, agent in enumerate(agents):
                    last_obs, reward, terminations, truncations, infos = env.last()
                    global_obs = env.state()
                    last_obs = torch.Tensor(partial_obs_extractor(last_obs, global_obs)).to(device)
                    action, logprob, _, value = agent.get_action_and_value(last_obs)
                    last_done = np.logical_or(terminations, truncations)
                    last_done = torch.Tensor([last_done.astype(int)]).to(device)

                    # store values
                    values[step, :, idagent] = value.flatten()
                    logprobs[step, :, idagent] = logprob
                    obs[step, :, idagent] = last_obs
                    dones[step, :, idagent] = last_done
                    rewards[step, :, idagent] = torch.tensor(reward).to(device).view(-1)
                    if "power" in infos:
                        powers.append(infos["power"])
                        writer.add_scalar(f"farm/power_T{idagent}", infos["power"], global_step)
                    if "load" in infos:
                        writer.add_scalar(f"farm/load_T{idagent}", sum(np.abs(infos["load"])), global_step)
                        writer.add_scalar(f"farm/controls/yaw_T{idagent}", last_obs[0].item(), global_step)

                    # store next action
                    env.step(make_yaw_action(action))
                    actions[step, :, idagent] = action
            
            writer.add_scalar(f"farm/power_total", sum(powers), global_step)
            writer.add_scalar(f"farm/reward", float(reward[0]), global_step)

        # bootstrap value for all agents and compute GAE
        lastgaelam = torch.zeros((args.num_envs, args.num_agents)).to(device)
        advantages = torch.zeros_like(rewards).to(device)
        with torch.no_grad():
            for t in reversed(range(0, args.num_steps)):
                nextvalues = values[t + 1]
                nextnonterminal = 1.0 - dones[t + 1]
                delta = rewards[t + 1] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs[:-1].reshape((-1,args.num_agents) + partial_obs_space.shape)
        b_logprobs = logprobs[:-1].reshape(-1, args.num_agents)
        b_actions = actions[:-1].reshape((-1, args.num_agents) + action_space.shape)
        b_advantages = advantages[:-1].reshape(-1, args.num_agents)
        b_returns = returns[:-1].reshape(-1, args.num_agents)
        b_values = values[:-1].reshape(-1, args.num_agents)

        # Optimizing the policy and value network

        for idagent, agent in enumerate(agents):
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds, idagent], b_actions[mb_inds, idagent])
                    logratio = newlogprob - b_logprobs[mb_inds, idagent]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds, idagent]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds, idagent]) ** 2
                        v_clipped = b_values[mb_inds, idagent] + torch.clamp(
                            newvalue - b_values[mb_inds, idagent],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds, idagent]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds, idagent]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizers[idagent].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizers[idagent].step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar(f"charts/agent_{idagent}/learning_rate", optimizers[idagent].param_groups[0]["lr"], global_step)
            writer.add_scalar(f"losses/agent_{idagent}/value_loss", v_loss.item(), global_step)
            writer.add_scalar(f"losses/agent_{idagent}/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar(f"losses/agent_{idagent}/entropy", entropy_loss.item(), global_step)
            writer.add_scalar(f"losses/agent_{idagent}/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar(f"losses/agent_{idagent}/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar(f"losses/agent_{idagent}/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar(f"losses/agent_{idagent}/explained_variance", explained_var, global_step)
        
            if args.save_model:
                model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
                for idagent, agent in enumerate(agents):
                    torch.save(agent.state_dict(), model_path+f"_{idagent}")
                print(f"model saved to {model_path}")
        
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
    env.close()
    writer.close()


    # Prepare plots
    fig = plot_env_history(env)
    fig.savefig(f"runs/{run_name}/plot.png")

    print("stop")








