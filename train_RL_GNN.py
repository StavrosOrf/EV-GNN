import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import wandb
import yaml
import random
import time
from tqdm import tqdm

import ev2gym
from ev2gym.rl_agent.reward import ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.reward import SimpleReward

from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads
from utils.state import PublicPST_GNN, V2G_ProfitMax_with_Loads_GNN

from utils.state import PublicPST_GNN_no_position_encoding, PublicPST_GNN_full_graph

from TD3.TD3_GNN import TD3_GNN
from TD3.TD3_ActionGNN import TD3_ActionGNN
from TD3.TD3 import TD3

from utils.replay_buffer import GNN_ReplayBuffer, ReplayBuffer, ActionGNN_ReplayBuffer

from SAC.sac import SAC
from SAC.actionSAC import SAC_ActionGNN

from utils.action_wrapper import BinaryAction, ThreeStep_Action

from gymnasium import Space
from torch_geometric.data import Data


class PyGDataSpace(Space):
    def __init__(self):
        super().__init__((), None)

    def sample(self):
        # Implement this method to generate a random Data object
        pass

    def contains(self, x):
        return isinstance(x, Data)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def eval_policy(policy, args, eval_episodes=30, config_file=None,discrete_actions=1):
    eval_env = gym.make('evs-v1')

    if discrete_actions == 3:
        eval_env = ThreeStep_Action(eval_env)    

    if "GNN" in args.policy:
        eval_env.observation_space = PyGDataSpace()

    avg_reward = 0.
    # use tqdm to show progress bar
    stats_list = []
    eval_stats = {}

    for _ in tqdm(range(eval_episodes)):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = policy.select_action(state, return_mapped_action=True)
            state, reward, done, _, stats = eval_env.step(action)
            avg_reward += reward

        stats_list.append(stats)

    # get the mean and std of the stats
    for key in stats.keys():
        eval_stats['eval_metrics/'+key +
                   '_mean'] = np.mean([x[key] for x in stats_list])
        eval_stats['eval_metrics/'+key +
                   '_std'] = np.std([x[key] for x in stats_list])

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward, eval_stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # TD3_GNN, TD3_ActionGNN, TD3, SAC, SAC_GNN, SAC_ActionGNN
    parser.add_argument("--policy", default="TD3_ActionGNN")
    parser.add_argument("--name", default="base")
    parser.add_argument("--project_name", default="ev2gym")
    parser.add_argument("--env", default="EV2Gym")
    # Config file for the environment: PublicPST.yaml, V2G_ProfixMaxWithLoads.yaml
    parser.add_argument("--config", default="PublicPST.yaml")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_timesteps", default=1e7, type=int)  # 1e7
    parser.add_argument("--load_model", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument('--group_name', type=str, default='')

    DEVELOPMENT = False

    if DEVELOPMENT:
        parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
        parser.add_argument("--eval_episodes", default=2, type=int)
        parser.add_argument("--start_timesteps", default=10,
                            type=int)
        parser.add_argument('--eval_freq', default=300, type=int)
        parser.add_argument("--batch_size", default=3, type=int)  # 256
        print(f'!!!!!!!!!!!!!!!! DEVELOPMENT MODE !!!!!!!!!!!!!!!!')
        print(f' Switch to production mode by setting DEVELOPMENT = False')
    else:
        parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
        parser.add_argument("--eval_episodes", default=100, type=int)
        parser.add_argument("--start_timesteps", default=2500,
                            type=int)  # original 25e5
        parser.add_argument("--eval_freq", default=2250,
                            type=int)  # in episodes
        parser.add_argument("--batch_size", default=256, type=int)  # 256

    parser.add_argument("--discount", default=0.99,
                        type=float)     # Discount factor
    # Target network update rate
    parser.add_argument("--tau", default=0.005, type=float)
    # Noise added to target policy during critic update

    # TD3 parameters #############################################
    parser.add_argument("--expl_noise", default=0.1, type=float)  # 0.1
    parser.add_argument("--policy_noise", default=0.2)  # 0.2
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--exp_prefix", default="")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--replay_buffer_size", default=1e6, type=int)

    # DT parameters #############################################
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--K', type=int, default=12)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)

    # SAC parameters #############################################
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--policy_SAC', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')

    # GNN Feature Extractor Parameters #############################################
    parser.add_argument('--fx_dim', type=int, default=8)
    parser.add_argument('--fx_GNN_hidden_dim', type=int, default=32)
    parser.add_argument('--fx_num_heads', type=int, default=2)
    parser.add_argument('--mlp_hidden_dim', type=int, default=256)
    parser.add_argument('--discrete_actions', type=int, default=1)
    parser.add_argument('--actor_num_gcn_layers', type=int, default=3)
    parser.add_argument('--critic_num_gcn_layers', type=int, default=3)
    
    
    parser.add_argument('--no_positional_encoding', type=bool, default=False)
    parser.add_argument('--full_graph', type=bool, default=False)
    
    scale = 1
    args = parser.parse_args()
    
    if args.discrete_actions > 1 and args.policy != "TD3_ActionGNN":
        raise ValueError(f"{args.policy} does not support discrete actions.")

    device = args.device
    device = device if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    replay_buffer_size = int(args.replay_buffer_size)

    config_file = f"./config_files/{args.config}"

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -")
    print(f'Config File: {config_file}')
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    if "V2G_ProfixMaxWithLoads" in config_file:
        args.group_name = "V2G_ProfixMaxWithLoads_"
        reward_function = ProfitMax_TrPenalty_UserIncentives
        if "GNN" in args.policy:
            state_function = V2G_ProfitMax_with_Loads_GNN
        else:
            state_function = V2G_profit_max_loads

    elif "PST" in config_file:
        if "GF" in config_file:
            args.group_name = "GF_PST_"
        else:
            args.group_name = "PST_"
        reward_function = SimpleReward
        if "GNN" in args.policy:
            if args.no_positional_encoding:
                state_function = PublicPST_GNN_no_position_encoding
                print("No positional encoding in state function.")
            elif args.full_graph:
                state_function = PublicPST_GNN_full_graph
                print("Full graph in state function.")
            else:
                state_function = PublicPST_GNN
                
            
        else:
            state_function = PublicPST

    config = yaml.load(open(config_file, 'r'),
                       Loader=yaml.FullLoader)
    gym.envs.register(id='evs-v1', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                      kwargs={'config_file': config_file,
                              'generate_rnd_game': True,
                              'reward_function': reward_function,
                              'state_function': state_function,
                              })

    env = gym.make('evs-v1')

    if args.discrete_actions == 3:
        env = ThreeStep_Action(env)
    elif args.discrete_actions != 1:
        raise ValueError("Discrete action number not recognized. Only support 1 or 3 at the moment!")

    if "GNN" in args.policy:
        env.observation_space = PyGDataSpace()

    global_target_return = 0

    exp_prefix = args.exp_prefix
    if exp_prefix != "":
        load_path = f"saved_models/{exp_prefix}"
    else:
        load_path = None

    # Set seeds
    # env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    simulation_length = config["simulation_length"]

    if "GF" in config_file:
        group_name = f'GF_{number_of_charging_stations}cs_{n_transformers}tr'
    else:
        if "SAC" in args.policy:
            group_name = f'{args.group_name}GNN_SAC_{number_of_charging_stations}cs_{n_transformers}tr'
        elif "TD3" in args.policy:
            group_name = f'{args.group_name}GNN_TD3_{number_of_charging_stations}cs_{n_transformers}tr'
        else:
            raise ValueError("Policy not recognized.")
        
    if args.discrete_actions > 1:
        group_name = "DiscreteActions_" + group_name

    if args.no_positional_encoding:
        group_name = "no_positional_encoding_" + group_name
        
    if args.full_graph:
        group_name = "full_graph_" + group_name
    
    exp_prefix = f'{args.name}-{random.randint(int(1e5), int(1e6) - 1)}'
    print(f'group_name: {group_name}, exp_prefix: {exp_prefix}')

    save_path = f'./saved_models/{exp_prefix}/'
    # create folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save config file
    with open(f'{save_path}/config.yaml', 'w') as file:
        yaml.dump(config, file)

    # np.save(f'{save_path}/state_mean.npy', state_mean.cpu().numpy())
    # np.save(f'{save_path}/state_std.npy', state_std.cpu().numpy())

    if args.log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project=args.project_name,
            save_code=True,
            config=config
        )

        wandb.run.log_code(".")

    kwargs = {
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "fx_dim": args.fx_dim,
        "fx_GNN_hidden_dim": args.fx_GNN_hidden_dim,
        "fx_num_heads": args.fx_num_heads,
        "actor_num_gcn_layers": args.actor_num_gcn_layers,
        "critic_num_gcn_layers": args.critic_num_gcn_layers,
    }

    # Initialize policy
    if args.policy == "TD3_GNN" or args.policy == "TD3_ActionGNN":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        kwargs["device"] = device
        kwargs["lr"] = args.lr

        kwargs['load_path'] = load_path
        kwargs['discrete_actions'] = args.discrete_actions
                
        # if statefunction has attribute node_sizes
        if hasattr(state_function, 'node_sizes'):
            kwargs['fx_node_sizes'] = state_function.node_sizes

        # Save kwargs to local path
        with open(f'{save_path}/kwargs.yaml', 'w') as file:
            yaml.dump(kwargs, file)

        if args.policy == "TD3_GNN":
            policy = TD3_GNN(**kwargs)
            replay_buffer = GNN_ReplayBuffer(action_dim=action_dim,
                                             max_size=replay_buffer_size,)
            # save the TD3_utils.py file using cp
            os.system(f'cp TD3/TD3_utils.py {save_path}')

        elif args.policy == "TD3_ActionGNN":
            policy = TD3_ActionGNN(**kwargs)
            replay_buffer = ActionGNN_ReplayBuffer(action_dim=action_dim,
                                                   max_size=replay_buffer_size,)
            os.system(f'cp TD3/TD3_Actionutils.py {save_path}')

    elif args.policy == "TD3":
        state_dim = env.observation_space.shape[0]
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        kwargs["device"] = device
        kwargs['state_dim'] = state_dim

        kwargs['load_path'] = load_path
        # Save kwargs to local path
        with open(f'{save_path}/kwargs.yaml', 'w') as file:
            yaml.dump(kwargs, file)

        policy = TD3(**kwargs)
        replay_buffer = ReplayBuffer(state_dim, action_dim)

    elif "SAC" in args.policy:

        kwargs["device"] = device
        kwargs["alpha"] = args.alpha
        kwargs["automatic_entropy_tuning"] = args.automatic_entropy_tuning
        kwargs["updates_per_step"] = args.updates_per_step
        kwargs["target_update_interval"] = args.target_update_interval
        kwargs["discount"] = args.discount
        kwargs["tau"] = args.tau
        kwargs['policy'] = args.policy_SAC
        kwargs['lr'] = args.lr
        kwargs['hidden_size'] = args.mlp_hidden_dim

        if hasattr(state_function, 'node_sizes'):
            fx_node_sizes = state_function.node_sizes

        if args.policy == "SAC_GNN":

            policy = SAC(num_inputs=-1,
                         action_space=env.action_space,
                         args=kwargs,
                         fx_node_sizes=fx_node_sizes,
                         GNN_fx=True)
            replay_buffer = GNN_ReplayBuffer(action_dim=action_dim,
                                             max_size=replay_buffer_size,)
            os.system(f'cp SAC/sac.py {save_path}')

        elif args.policy == "SAC_ActionGNN":
            policy = SAC_ActionGNN(action_space=env.action_space,
                                   fx_node_sizes=fx_node_sizes,
                                   args=kwargs,)
            replay_buffer = ActionGNN_ReplayBuffer(action_dim=action_dim,
                                                   max_size=replay_buffer_size,)
            os.system(f'cp SAC/actionSAC.py {save_path}')

        elif args.policy == "SAC":
            state_dim = env.observation_space.shape[0]
            policy = SAC(num_inputs=state_dim,
                         action_space=env.action_space,
                         args=kwargs)
            replay_buffer = ReplayBuffer(state_dim, action_dim)
            os.system(f'cp SAC/sac.py {save_path}')

    else:
        raise ValueError("Policy not recognized.")

    # save kwargs to save_path
    with open(f'{save_path}/kwargs.yaml', 'w') as file:
        yaml.dump(kwargs, file)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    print(
        f'action_dim: {action_dim}, replay_buffer_size: {replay_buffer_size}')
    print(f'max_episode_length: {simulation_length}')

    # Evaluate untrained policy

    evaluations = []
    episode_num = -1
    t = 0
    updates = 0
    best_reward = -np.Inf
    episode_timesteps = 0
    episode_reward = 0

    state, _ = env.reset()
    ep_start_time = time.time()

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps and args.policy != "TD3_ActionGNN" and args.policy != "SAC_ActionGNN":
            action = env.action_space.sample()
            next_state, reward, done, _, stats = env.step(action)
        else:

            if args.policy == "TD3_ActionGNN":
                mapped_action, action = policy.select_action(
                    state, expl_noise=args.expl_noise)

                # Perform action
                next_state, reward, done, _, stats = env.step(mapped_action)

            elif args.policy == "SAC_ActionGNN":
                mapped_action, action = policy.select_action(state,
                                                             evaluate=False,
                                                             return_mapped_action=True)

                # Perform action
                next_state, reward, done, _, stats = env.step(mapped_action)
            elif "SAC" in args.policy:
                action = policy.select_action(state, evaluate=False)
                # Perform action
                next_state, reward, done, _, stats = env.step(action)

            elif args.policy == "TD3" or args.policy == "TD3_GNN":
                # Select action randomly or according to policy + add noise
                action = (
                    policy.select_action(state)
                    + np.random.normal(0, max_action *
                                       args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
                # Perform action
                next_state, reward, done, _, stats = env.step(action)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, float(done))

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:

            start_time = time.time()
            if 'SAC' in args.policy:
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = policy.train(
                    replay_buffer, args.batch_size, updates)
                updates += 1

                if args.log_to_wandb:
                    wandb.log({'train/critic_loss': critic_1_loss,
                               'train/critic2_loss': critic_2_loss,
                               'train/actor_loss': policy_loss,
                               'train/ent_loss': ent_loss,
                               'train/alpha': alpha,
                               'train/time': time.time() - start_time, })

            else:
                critic_loss, actor_loss = policy.train(
                    replay_buffer, args.batch_size)

                if args.log_to_wandb and actor_loss is not None:
                    wandb.log({'train/critic_loss': critic_loss,
                               'train/actor_loss': actor_loss,
                               'train/time': time.time() - start_time, })

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}" +
                f" Time: {time.time() - ep_start_time:.3f}")
            # Reset environment
            state, _ = env.reset()
            ep_start_time = time.time()
            done = False

            episode_num += 1

            if args.log_to_wandb:
                wandb.log({'train_ep/episode_reward': episode_reward,
                           'train_ep/episode_num': episode_num})

            episode_reward = 0
            episode_timesteps = 0

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:

            avg_reward, eval_stats = eval_policy(policy,
                                                 args=args,
                                                 eval_episodes=args.eval_episodes,
                                                 config_file=config_file,
                                                 discrete_actions=args.discrete_actions)
            evaluations.append(avg_reward)

            if evaluations[-1] > best_reward:
                best_reward = evaluations[-1]

                policy.save(f'saved_models/{exp_prefix}/model.best')

            if args.log_to_wandb:
                wandb.log({'eval/mean_reward': evaluations[-1],
                           'eval/best_reward': best_reward, })

                wandb.log(eval_stats)

    if args.log_to_wandb:
        wandb.finish()

    policy.save(f'saved_models/{exp_prefix}/model.last')
    avg_reward, eval_stats = eval_policy(policy,
                                         args=args,
                                         eval_episodes=args.eval_episodes,
                                         config_file=config_file,
                                         discrete_actions=args.discrete_actions)
    print(f'Final Evaluation: {avg_reward}')
