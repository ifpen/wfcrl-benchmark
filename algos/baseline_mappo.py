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
from utils import (
    LocalSummaryWriter,
    eval_wind_rose,
    eval_policies, 
    plot_env_history, 
    prepare_eval_windrose, 
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    scenario: str = "constant"
    """the name of the scenario, identified by the wind series"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    debug_log: bool = False
    """if toggled, will log all power outputs and yaws step by step"""
    wandb_project_name: str = "benchmark-wfcrl-v2"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    wind_data: str = "data/smarteole.csv"
    """Path to wind data for wind rose evaluation"""
    load_coef: float = 1
    """coefficient of the load penalty"""
    episode_length: int = 150
    """side of an trajectory to store in buffer size"""

    # Algorithm specific arguments
    env_id: str = "Dec_Turb3_Row1_Floris"
    """the id of the environment"""
    total_timesteps: int = int(1e5)
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

    # DFAC arguments
    pretrained_models: str = None
    """Path to pretrained models"""
    policy: str = "base"
    """Type of policy"""
    hidden_layer_nn: Union[bool, Union[bool, tuple[int, ...]]] = (64, 64)
    """number of neurons in hidden layer"""
    num_steps: int = 2048
    """number of available rewards before update"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    freq_eval: int = 5
    """Number of iterations between eval"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    num_agents: int = 1
    """the number of agents in the environment"""
    reward_shaping: str = ""
    """Toggle learning rate annealing for policy and value networks"""
    device: str = "cpu"
    "device"

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
class SharedCritic(nn.Module):
    def __init__(self, observation_space, hidden_layers):
        super().__init__()
        self.observation_space = observation_space
        input_layers = [np.prod(observation_space.shape)] + list(hidden_layers)
        self.critic = nn.Sequential(
            *[
                nn.Sequential(layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh())
                for in_dim, out_dim in zip(input_layers[:-1], input_layers[1:])
            ],
            layer_init(nn.Linear(input_layers[-1], 1), std=1.0),
        )
        self.register_buffer(
            "observation_low", torch.tensor(observation_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "observation_high", torch.tensor(observation_space.high, dtype=torch.float32)
        )

    def get_value(self, x):
        x = (x - self.observation_low)/(self.observation_high - self.observation_low)
        return self.critic(x)

class Agent(nn.Module):
    def __init__(self, observation_space, action_space, hidden_layers):
        super().__init__()
        action_dim = np.prod(action_space.shape)

        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        self.observation_space = observation_space

        input_layers = [np.prod(observation_space.shape)] + list(hidden_layers)
        self.actor = nn.Sequential(
            *[
                nn.Sequential(layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh())
                for in_dim, out_dim in zip(input_layers[:-1], input_layers[1:])
            ],
            layer_init(nn.Linear(input_layers[-1], action_dim), std=1.0),
        )
        self.register_buffer(
            "observation_low", torch.tensor(observation_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "observation_high", torch.tensor(observation_space.high, dtype=torch.float32)
        )

    def get_action(self, x, action=None, deterministic=False):
        x = (x - self.observation_low)/(self.observation_high - self.observation_low)
        action_mean = self.actor(x)
        action_std = torch.ones_like(action_mean) * self.log_std.exp()
        distribution = Normal(action_mean, action_std)
        if action is None:
            action = distribution.mode if deterministic else distribution.rsample()
        return action, distribution.log_prob(action).sum(-1), distribution.entropy()


if __name__ == "__main__":
    args = tyro.cli(Args)
    controls = ["yaw"]
    assert args.scenario in ["constant", "windrose"]
    args.batch_size = int(args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    num_episodes = int(np.ceil(args.num_steps / (args.episode_length-1)))
    env = envs.make(
        args.env_id,
        controls=controls, 
        max_num_steps=args.episode_length,
        load_coef=args.load_coef
    )
    args.num_agents = env.num_turbines
    args.reward_shaping = ""
    run_name = f"{args.env_id}__{args.exp_name}__{args.scenario}_{args.seed}__{int(time.time())}"
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
    global_obs_extractor = VectorExtractor(env.state_space)
    partial_obs_space = partial_obs_extractor.space
    global_obs_space = global_obs_extractor.space
    action_space = action_space_extractor.space
    hidden_layer_nn = [] if not isinstance(args.hidden_layer_nn, tuple) else args.hidden_layer_nn
    agents = [
        Agent(partial_obs_space, action_space, hidden_layer_nn).to(device)
        for _ in range(args.num_agents)
    ]
    shared_critic = SharedCritic(global_obs_space, hidden_layer_nn).to(device)
    actor_optimizers = [
        optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        for agent in agents
    ]
    critic_optimizer = optim.Adam(shared_critic.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.pretrained_models is not None:
        args.pretrained_models = Path(args.pretrained_models)
        assert args.pretrained_models.exists()
        for idagent, agent in enumerate(agents):
            try:
                path = list(args.pretrained_models.glob(f"*model_{idagent}"))[0]
            except:
                raise FileNotFoundError(f"No file in model_{idagent} found under folder {args.pretrained_models}")
            params = torch.load(str(path), map_location='cpu')
            agent.load_state_dict(params)
    
    def get_deterministic_action(agent, observation, last_action=None):
        observation = torch.Tensor(partial_obs_extractor(observation)).to(device)
        action, _, _ = agent.get_action(observation, deterministic=True)
        return action_space_extractor.make_dict(action)
    
    windrose_eval = args.scenario == "windrose"
    if windrose_eval:
        windrose = prepare_eval_windrose(args.wind_data, num_bins=5)
        def evaluate(eval_env, eval_agents):
            return eval_wind_rose(eval_env, eval_agents, windrose, get_deterministic_action)[0]
        env.reset(args.seed)
    else:
        def evaluate(eval_env, eval_agents):
            return eval_policies(eval_env, eval_agents, get_deterministic_action)
        env.reset(options={"wind_speed": 8, "wind_direction": 270})

    # ALGO Logic: Storage setup
    global_obs = torch.zeros((args.episode_length, args.num_steps) + global_obs_space.shape).to(device)
    obs = torch.zeros((args.episode_length, num_episodes, args.num_agents) + partial_obs_space.shape).to(device)
    actions = torch.zeros((args.episode_length, num_episodes, args.num_agents) + action_space.shape).to(device)
    logprobs = torch.zeros((args.episode_length, num_episodes, args.num_agents)).to(device)
    rewards = torch.zeros((args.episode_length, num_episodes, args.num_agents)).to(device)
    dones = torch.zeros((args.episode_length, num_episodes, args.num_agents)).to(device)
    terminations = torch.zeros((args.episode_length, num_episodes, args.num_agents)).to(device)
    values = torch.zeros((args.episode_length, num_episodes, args.num_agents)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()    

    for iteration in range(args.num_iterations):
        if iteration % args.freq_eval  == 0:
            print(f"Evaluating at iteration {iteration}")
            eval_score = evaluate(env, agents)
            print("Episode:", iteration, "Score: ", eval_score)
            writer.add_scalar(f"eval/eval_score", eval_score, global_step)

        last_done = False
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - iteration / args.num_iterations
            lrnow = frac * args.learning_rate
            for optimizer in actor_optimizers + [critic_optimizer]:
                optimizer.param_groups[0]["lr"] = lrnow

        step = pt = episode_id = 0
        while step < args.num_steps:
            if last_done:
                episode_id += 1
                pt = 0
                writer.add_scalar(f"farm/episode_reward", float(cumul_rewards), global_step)
                writer.add_scalar(f"farm/episode_power", float(cumul_power) / args.episode_length, global_step)
                writer.add_scalar(f"farm/episode_load", float(cumul_load) / args.episode_length, global_step)
            if last_done or step == 0:
                if windrose_eval:
                    env.reset(args.seed+global_step)
                else:
                    env.reset(options={"wind_speed": 8, "wind_direction": 270})
                cumul_rewards = cumul_power = cumul_load = 0
            global_step += 1
            powers = []
            loads = []
            with torch.no_grad():
                last_global_obs = env.state()
                last_global_obs = torch.tensor(global_obs_extractor(last_global_obs), dtype=torch.float32, device=device)
                value = shared_critic.get_value(last_global_obs)
                global_obs[pt, episode_id, :] = last_global_obs
                values[pt, episode_id, :] = value.flatten()
                for idagent, agent in enumerate(agents):
                    last_obs, reward, termination, truncation, infos = env.last()
                    last_obs = torch.tensor(partial_obs_extractor(last_obs), dtype=torch.float32, device=device)
                    action, logprob, _ = agent.get_action(last_obs)
                    last_done = np.logical_or(termination, truncation)
                    
                    # store values
                    logprobs[pt, episode_id, idagent] = logprob
                    obs[pt, episode_id, idagent] = last_obs
                    dones[pt, episode_id, idagent] = torch.tensor([last_done.astype(int)], device=device)
                    rewards[pt, episode_id, idagent] = torch.tensor(reward, dtype=torch.float32, device=device).view(-1)
                    actions[pt, episode_id, idagent] = action
                    if "power" in infos:
                        powers.append(infos["power"])
                    if "load" in infos:
                        loads.append(float(np.mean(np.abs(infos["load"]))))
                    if args.debug_log:
                        if "power" in infos:
                            writer.add_scalar(f"farm/power_T{idagent}", infos["power"], global_step)
                        if "load" in infos:
                            writer.add_scalar(f"farm/load_T{idagent}", sum(np.abs(infos["load"])), global_step)
                        writer.add_scalar(f"farm/controls/yaw_T{idagent}", last_obs[0].item(), global_step)
                        writer.add_scalar(f"charts/action/yaw_T{idagent}", action[0], global_step)

                    if last_done:
                        env.step(None)
                    else:
                        env.step(action_space_extractor.make_dict(action))
                    
            if args.debug_log:
                writer.add_scalar(f"farm/reward", float(reward[0]), global_step)  
                if "power" in infos:
                    writer.add_scalar(f"farm/power_total", sum(powers), global_step)
            step += int(not last_done)
            pt += 1
            cumul_power += sum(powers)
            cumul_load += sum(loads)
            cumul_rewards += float(reward[0])

        # normalize rewards
        rewards_mean = rewards[:, :, 0].flatten()[:-(args.episode_length - pt)].mean()
        rewards_std = rewards[:, :, 0].flatten()[:-(args.episode_length - pt)].std()
        rewards = (rewards - rewards_mean) / (rewards_std + 1e-8)

        # bootstrap value for all agents and compute GAE
        dones[pt-1, episode_id, :] = 1
        lastgaelam = torch.zeros((num_episodes, args.num_agents)).to(device)
        advantages = torch.zeros_like(rewards).to(device)
        with torch.no_grad():
            for t in reversed(range(0, args.episode_length-1)):
                nextvalues = values[t + 1]
                nextnonterminal = 1.0 - terminations[t + 1]
                nextbootstrap = 1.0 - dones[t + 1]
                delta = rewards[t + 1] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextbootstrap * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_global_obs = global_obs[:-1].transpose(0,1).reshape((-1,) + global_obs_space.shape)[:args.batch_size]
        b_obs = obs[:-1].transpose(0,1).reshape((-1,args.num_agents) + partial_obs_space.shape)[:args.batch_size]
        b_logprobs = logprobs[:-1].transpose(0,1).reshape(-1, args.num_agents)[:args.batch_size]
        b_actions = actions[:-1].transpose(0,1).reshape((-1, args.num_agents) + action_space.shape)[:args.batch_size]
        b_advantages = advantages[:-1].transpose(0,1).reshape(-1, args.num_agents)[:args.batch_size]
        b_returns = returns[:-1].transpose(0,1).reshape(-1, args.num_agents)[:args.batch_size]
        b_values = values[:-1].transpose(0,1).reshape(-1, args.num_agents)[:args.batch_size]

        # Optimizing the policy and value network

        for epoch in range(args.update_epochs):
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                for idagent, agent in enumerate(agents):
                    _, newlogprob, entropy = agent.get_action(b_obs[mb_inds, idagent], b_actions[mb_inds, idagent])
                    logratio = newlogprob - b_logprobs[mb_inds, idagent]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds, idagent]
                    if args.norm_adv:
                        # mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                        mb_advantages = (mb_advantages - b_advantages[:, idagent].mean()) / (b_advantages[:, idagent].std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss

                    actor_optimizers[idagent].zero_grad()
                    pg_loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    actor_optimizers[idagent].step()

                    if (iteration % 5 == 0) and (epoch == args.update_epochs-1) and (end >= args.batch_size-1):
                        writer.add_scalar(f"charts/agent_{idagent}/learning_rate", actor_optimizers[idagent].param_groups[0]["lr"], global_step)
                        writer.add_scalar(f"losses/agent_{idagent}/policy_loss", pg_loss.item(), global_step)
                        writer.add_scalar(f"losses/agent_{idagent}/entropy", entropy_loss.item(), global_step)
                        writer.add_scalar(f"losses/agent_{idagent}/old_approx_kl", old_approx_kl.item(), global_step)
                        writer.add_scalar(f"losses/agent_{idagent}/approx_kl", approx_kl.item(), global_step)
                        writer.add_scalar(f"losses/agent_{idagent}/clipfrac", np.mean(clipfracs), global_step)
                
                # Global Value loss
                # global_obs = b_obs[mb_inds, :].view(args.minibatch_size, -1)
                newvalue = shared_critic.get_value(b_global_obs[mb_inds])
                newvalue = newvalue.view(-1)

                if args.clip_vloss:
                    # the return/value is the same for all agents
                    v_loss_unclipped = (newvalue - b_returns[mb_inds, 0]) ** 2
                    v_clipped = b_values[mb_inds, 0] + torch.clamp(
                        newvalue - b_values[mb_inds, 0],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds, 0]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds, 0]) ** 2).mean()

                global_v_loss = v_loss * args.vf_coef
                critic_optimizer.zero_grad()
                global_v_loss.backward()
                nn.utils.clip_grad_norm_(shared_critic.parameters(), args.max_grad_norm)
                critic_optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        if iteration % 5 == 0:
            writer.add_scalar(f"losses/value_loss", global_v_loss.item(), global_step)
            writer.add_scalar(f"losses/explained_variance", explained_var, global_step)
    
        if (iteration % 100 == 0) and args.save_model:
            for idagent, agent in enumerate(agents):
                torch.save(agent.state_dict(), model_path+f"_{idagent}")
            torch.save(shared_critic.state_dict(), model_path+f"_critic")
            print(f"model saved to {model_path}")
        
            # print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
    env.close()
    for idagent, agent in enumerate(agents):
        torch.save(agent.state_dict(), model_path+f"_{idagent}")
    torch.save(shared_critic.state_dict(), model_path+f"_critic")
    print(f"model saved to {model_path}")
    writer.close()


    # Prepare plots
    fig = plot_env_history(env)
    fig.savefig(f"runs/{run_name}/plot.png")