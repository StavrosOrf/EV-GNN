# This script reads the replay files and evaluates the performance.

import yaml
import os
import pickle
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import datetime
import time

from ev2gym.models.ev2gym_env import EV2Gym

from ev2gym.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible
from ev2gym.baselines.heuristics import ChargeAsFastAsPossibleToDesiredCapacity
from ev2gym.baselines.heuristics import RoundRobin_GF_off_allowed, RoundRobin_GF

from ev2gym.baselines.mpc.ocmf_mpc import OCMF_V2G, OCMF_G2V
from ev2gym.baselines.mpc.eMPC import eMPC_V2G, eMPC_G2V
from ev2gym.baselines.mpc.eMPC_v2 import eMPC_V2G_v2

from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle, V2GProfitMaxLoadsOracle

from stable_baselines3 import PPO, A2C, DDPG, SAC
from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO


# GNN-based models evaluations
from TD3.TD3_GNN import TD3_GNN
from TD3.TD3_ActionGNN import TD3_ActionGNN
from TD3.old_TD3_ActionGNN import TD3_ActionGNN as TD3_ActionGNN_old

from TD3.TD3 import TD3
from SAC.sac import SAC
from SAC.actionSAC import SAC_ActionGNN
from SAC.old_actionSAC import SAC_ActionGNN as SAC_ActionGNN_old
from GNN.state import PublicPST_GNN, V2G_ProfitMax_with_Loads_GNN


# from DT.evaluation.evaluate_episodes import evaluate_episode_rtg_from_replays
# from DT.models.decision_transformer import DecisionTransformer
# from DT.load_model import load_DT_model

from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin
from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB

from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, SimpleReward
from ev2gym.rl_agent.reward import profit_maximization, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads

from ev2gym.visuals.evaluator_plot import plot_total_power, plot_comparable_EV_SoC
from ev2gym.visuals.evaluator_plot import plot_total_power_V2G, plot_actual_power_vs_setpoint
from ev2gym.visuals.evaluator_plot import plot_comparable_EV_SoC_single, plot_prices

from state_action_eda import AnalysisReplayBuffer

import gymnasium as gym
import torch


def evaluator():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ############# Simulation Parameters #################
    n_test_cycles = 1
    SAVE_REPLAY_BUFFER = False
    SAVE_EV_PROFILES = False

    # config_file = "./config_files/PublicPST.yaml"  # 25
    # config_file = "./config_files/PublicPST_100.yaml"
    # config_file = "./config_files/PublicPST_500.yaml"
    config_file = "./config_files/PublicPST_1000.yaml"

    # config_file = "./config_files/PublicPST_100_g1.yaml"
    # config_file = "./config_files/PublicPST_100_g2.yaml"
    # config_file = "./config_files/PublicPST_100_g3.yaml"

    # config_file = "./config_files/PublicPST_500_g1.yaml"
    # config_file = "./config_files/PublicPST_500_g2.yaml"
    # config_file = "./config_files/PublicPST_500_g3.yaml"

    # config_file = "./config_files/GF_PST_25.yaml"
    # config_file = "./config_files/GF_PST_100.yaml"
    # config_file = "./config_files/GF_PST_500.yaml"

    # config_file = "./config_files/V2G_ProfixMaxWithLoads_25.yaml"
    # config_file = "./config_files/V2G_ProfixMaxWithLoads_100.yaml"
    # config_file = "./config_files/V2G_ProfixMaxWithLoads_500.yaml"

    if "V2G_ProfixMaxWithLoads" in config_file:
        state_function_Normal = V2G_profit_max_loads
        state_function_GNN = V2G_ProfitMax_with_Loads_GNN
        reward_function = profit_maximization

    elif "PST" in config_file:
        state_function_Normal = PublicPST
        state_function_GNN = PublicPST_GNN
        reward_function = SimpleReward
    else:
        raise ValueError(f'Unknown config file {config_file}')

    # Algorithms to compare:
    # Use algorithm name or the saved RL model path as string
    algorithms = [
        # ChargeAsFastAsPossible,
        # DecisionTransformer,
        # ChargeAsLateAsPossible,
        # PPO, A2C, DDPG, SAC, TD3, TQC, TRPO, ARS, RecurrentPPO,
        # SAC,
        # TQC,
        # # TD3,
        # # ARS,
        # # RecurrentPPO,

        # ----------------------------------------------------------------

        ### V2G_ProfixMaxWithLoads 25CS
        # "TD3_run_3_29306-404683",
        # "TD3_GNN_run_3_24008-870281",
        # "TD3_ActionGNN_run_3_36407-816505",
        # "SAC_GNN_run_1_65284-849912",
        # "SAC_ActionGNN_run_2_31150-671107",
        # "SAC_run_1_11569-249914",

        ### V2G_ProfixMaxWithLoads 500CS
        # "SAC_GNN_run_2_96519-630780",
        # "SAC_run_0_33667-400451",
        # "SAC_ActionGNN_run_0_82569-619094",
        # "TD3_run_3_71992-180065",
        # "TD3_GNN_run_1_4841-176458",
        # "TD3_ActionGNN_run_3_9592-170267",

        # ----------------------------------------------------------------

        ### PST 25CS
        # "SAC_run_0_25377-546286",
        # "SAC_GNN_run_0_6651-167314",
        # "SAC_ActionGNN_run_0_39335-597033",
        # "TD3_ActionGNN_run_3_99695-870291",
        # "TD3_GNN_run_2_25852-556176",
        # "TD3_run_0_47448-558478",

        ### PST 100cs
        # "SAC_ActionGNN_run_2_57591-144641",
        # "SAC_GNN_run_0_16099-876950",
        # "SAC_run_2_25877-194576",
        # "TD3_run_4_94486-719210",
        # "TD3_GNN_run_4_14896-733342",
        # "TD3_ActionGNN_run_2_31792-403621",

        ### 500cs PST
        # "TD3_ActionGNN_run_1_83995-459455",
        # "TD3_GNN_run_0_61640-574718",
        # "TD3_run_1_571-641701",
        # "SAC_run_3_43478-600660",
        # "SAC_ActionGNN_run_0_91588-915095",
        # "SAC_GNN_run_0_36109-236450",

        # # 1000cs PST
        "TD3_GNN_run_2_21249-577916",
        "TD3_run_3_88194-314562",
        "SAC_GNN_run_0_78576-537380",
        "SAC_run_3_21562-252852",
        "SAC_ActionGNN_run_0_48341-402851",
        "TD3_ActionGNN_run_0_37311-723077",
        # # --------------------------------
        "tqc_run_1_54112-161532",
        "ppo_run_0_41272-573476",
        "ddpg_run_1_94166-370538",
        "trpo_run_0_97057-944000",
        "a2c_run_0_35840-104916",


        # ----------------------------------------------------------------

        # RoundRobin_GF_off_allowed,
        # RoundRobin_GF,
        # RoundRobin,

        # eMPC_V2G,
        # eMPC_V2G_v2,
        # # V2GProfitMaxLoadsOracle,
        # V2GProfitMaxOracleGB,
        # V2GProfitMaxOracle,
        # PowerTrackingErrorrMin
    ]

    # create a AnalysisReplayBuffer object for each algorithm

    env = EV2Gym(config_file=config_file,
                 generate_rnd_game=True,
                 state_function=state_function_Normal,
                 reward_function=reward_function,
                 )

    if SAVE_REPLAY_BUFFER:
        replay_buffers = {}
        for algorithm in algorithms:

            replay_buffers[algorithm] = AnalysisReplayBuffer(state_dim=env.observation_space.shape[0],
                                                             action_dim=env.action_space.shape[0],
                                                             max_size=int(1e4))

    #####################################################

    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    timescale = config["timescale"]
    simulation_length = config["simulation_length"]

    scenario = config_file.split("/")[-1].split(".")[0]
    eval_replay_path = f'./replay/{number_of_charging_stations}cs_{n_transformers}tr_{scenario}/'
    print(f'Looking for replay files in {eval_replay_path}')
    try:
        eval_replay_files = [f for f in os.listdir(
            eval_replay_path) if os.path.isfile(os.path.join(eval_replay_path, f))]

        print(
            f'Found {len(eval_replay_files)} replay files in {eval_replay_path}')
        if n_test_cycles > len(eval_replay_files):
            n_test_cycles = len(eval_replay_files)

        replay_to_print = 1
        replay_to_print = min(replay_to_print, len(eval_replay_files)-1)
        replays_exist = True

    except:
        n_test_cycles = n_test_cycles
        replays_exist = False

    print(f'Number of test cycles: {n_test_cycles}')

    if SAVE_EV_PROFILES:
        ev_profiles = []

    def generate_replay(evaluation_name):
        env = EV2Gym(config_file=config_file,
                     generate_rnd_game=True,
                     save_replay=True,
                     replay_save_path=f"replay/{evaluation_name}/",
                     )
        replay_path = f"replay/{evaluation_name}/replay_{env.sim_name}.pkl"

        for _ in range(env.simulation_length):
            actions = np.ones(env.cs)

            new_state, reward, done, truncated, _ = env.step(
                actions, visualize=False)  # takes action

            if done:
                break

        if SAVE_EV_PROFILES:
            ev_profiles.append(env.EVs_profiles)
        return replay_path

    evaluation_name = f'eval_{number_of_charging_stations}cs_{n_transformers}tr_{scenario}_{len(algorithms)}_algos' +\
        f'_{n_test_cycles}_exp_' +\
        f'{datetime.datetime.now().strftime("%Y_%m_%d_%f")}'

    # make a directory for the evaluation
    save_path = f'./results/{evaluation_name}/'
    os.makedirs(save_path, exist_ok=True)
    os.system(f'cp {config_file} {save_path}')

    if not replays_exist:
        eval_replay_files = [generate_replay(
            evaluation_name) for _ in range(n_test_cycles)]

    # save the list of EV profiles to a pickle file
    if SAVE_EV_PROFILES:
        with open(save_path + 'ev_profiles.pkl', 'wb') as f:
            print(f'Saving EV profiles to {save_path}ev_profiles.pkl')
            pickle.dump(ev_profiles, f)

        exit()

    plot_results_dict = {}
    counter = 0
    for algorithm in algorithms:
        print(' +------- Evaluating', algorithm, " -------+")
        for k in range(n_test_cycles):
            print(f' Test cycle {k+1}/{n_test_cycles} -- {algorithm}')
            counter += 1
            h = -1

            if replays_exist:
                replay_path = eval_replay_path + eval_replay_files[k]
            else:
                replay_path = eval_replay_files[k]

            if type(algorithm) == str:
                if "GNN" in algorithm:
                    state_function = state_function_GNN
                else:
                    state_function = state_function_Normal
            else:
                state_function = state_function_Normal

            env = EV2Gym(config_file=config_file,
                         load_from_replay_path=replay_path,
                         state_function=state_function,
                         reward_function=reward_function,
                         )

            # initialize the timer
            timer = time.time()
            state, _ = env.reset()
            try:
                if type(algorithm) == str:
                    if algorithm.split('_')[0] in ['OCMF', 'eMPC']:
                        h = int(algorithm.split('_')[2])
                        algorithm = algorithm.split(
                            '_')[0] + '_' + algorithm.split('_')[1]
                        print(
                            f'Algorithm: {algorithm} with control horizon {h}')
                        if algorithm == 'OCMF_V2G':
                            model = OCMF_V2G(env=env, control_horizon=h)
                            algorithm = OCMF_V2G
                        elif algorithm == 'OCMF_G2V':
                            model = OCMF_G2V(env=env, control_horizon=h)
                            algorithm = OCMF_G2V
                        elif algorithm == 'eMPC_V2G':
                            model = eMPC_V2G(env=env, control_horizon=h)
                            algorithm = eMPC_V2G
                        elif algorithm == 'eMPC_G2V':
                            model = eMPC_G2V(env=env, control_horizon=h)
                            algorithm = eMPC_G2V

                        algorithm_name = algorithm.__name__

                    elif any(algo in algorithm for algo in ['ppo', 'a2c', 'ddpg', 'tqc', 'trpo', 'ars', 'rppo']):

                        gym.envs.register(id='evs-v0', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                                          kwargs={'config_file': config_file,
                                                  'generate_rnd_game': True,
                                                  'state_function': state_function_Normal,
                                                  'reward_function': reward_function,
                                                  'load_from_replay_path': replay_path,
                                                  })
                        env = gym.make('evs-v0')

                        load_path = f'./eval_models/{algorithm}/best_model.zip'

                        # initialize the timer
                        timer = time.time()
                        algorithm_name = algorithm.split('_')[0]

                        if 'rppo' in algorithm:
                            sb3_algo = RecurrentPPO
                        elif 'ppo' in algorithm:
                            sb3_algo = PPO
                        elif 'a2c' in algorithm:
                            sb3_algo = A2C
                        elif 'ddpg' in algorithm:
                            sb3_algo = DDPG
                        elif 'tqc' in algorithm:
                            sb3_algo = TQC
                        elif 'trpo' in algorithm:
                            sb3_algo = TRPO
                        else:
                            exit()

                        model = sb3_algo.load(load_path,
                                              env,
                                              device=device)
                        # set replay buffer to None

                        if 'tqc' in algorithm or 'ddpg' in algorithm:
                            model.replay_buffer = model.replay_buffer.__class__(1,
                                                                                model.observation_space,
                                                                                model.action_space,
                                                                                device=model.device,
                                                                                optimize_memory_usage=model.replay_buffer.optimize_memory_usage)

                        env = model.get_env()
                        state = env.reset()

                    elif "SAC" in algorithm:
                        load_model_path = f'./eval_models/{algorithm}/'
                        # Load kwargs.yaml as a dictionary
                        with open(f'{load_model_path}kwargs.yaml') as file:
                            kwargs = yaml.load(
                                file, Loader=yaml.FullLoader)

                        if hasattr(state_function, 'node_sizes'):
                            fx_node_sizes = state_function.node_sizes

                        if "ActionGNN" in algorithm:
                            algorithm_name = "SAC_ActionGNN"
                            if 'actor_num_gcn_layers' in kwargs:
                                model = SAC_ActionGNN(action_space=env.action_space,
                                                      fx_node_sizes=fx_node_sizes,
                                                      args=kwargs,)
                            else:
                                model = SAC_ActionGNN_old(action_space=env.action_space,
                                                          fx_node_sizes=fx_node_sizes,
                                                          args=kwargs,)

                            if "extraLayers" in algorithm:
                                state_dict = torch.load(f'{load_model_path}model.best')[
                                    'policy_state_dict']
                                new_state_dict = {}
                                rename_map = {
                                    "gcn_conv2.bias": "gcn_layers.0.bias",
                                    "gcn_conv2.lin.weight": "gcn_layers.0.lin.weight",
                                    "gcn_conv2_1.bias": "gcn_layers.1.bias",
                                    "gcn_conv2_1.lin.weight": "gcn_layers.1.lin.weight",
                                    "gcn_conv2_2.bias": "gcn_layers.2.bias",
                                    "gcn_conv2_2.lin.weight": "gcn_layers.2.lin.weight",
                                }

                                for old_key, new_key in rename_map.items():
                                    if old_key in state_dict:
                                        new_state_dict[new_key] = state_dict[old_key]
                                # Now we need to check if there are other keys in the state_dict that do not need renaming
                                for key in state_dict:
                                    if key not in rename_map:
                                        new_state_dict[key] = state_dict[key]

                                model.policy.load_state_dict(
                                    new_state_dict)
                            else:
                                model.load(ckpt_path=f'{load_model_path}model.best',
                                           evaluate=True)

                        elif "GNN" in algorithm:
                            model = SAC(num_inputs=-1,
                                        action_space=env.action_space,
                                        args=kwargs,
                                        fx_node_sizes=fx_node_sizes,
                                        GNN_fx=True)

                            algorithm_name = "SAC_GNN"
                            model.load(ckpt_path=f'{load_model_path}model.best',
                                       evaluate=True)
                        else:
                            state_dim = env.observation_space.shape[0]
                            model = SAC(num_inputs=state_dim,
                                        action_space=env.action_space,
                                        args=kwargs)

                            algorithm_name = "SAC"
                            model.load(ckpt_path=f'{load_model_path}model.best',
                                       evaluate=True)

                        if k == 0:
                            actor_model = model.policy
                            model_parameters = filter(
                                lambda p: p.requires_grad, actor_model.parameters())
                            params = sum([np.prod(p.size())
                                          for p in model_parameters])
                            print(
                                f'Actor model has {params} trainable parameters')

                    elif "TD3" in algorithm:
                        load_model_path = f'./eval_models/{algorithm}/'
                        # Load kwargs.yaml as a dictionary
                        with open(f'{load_model_path}kwargs.yaml') as file:
                            kwargs = yaml.load(
                                file, Loader=yaml.FullLoader)

                        if "ActionGNN" in algorithm:
                            algorithm_name = "TD3_ActionGNN"
                            print("Loading TD3_ActionGNN")
                            if 'actor_num_gcn_layers' in kwargs:
                                model = TD3_ActionGNN(**kwargs)
                            else:
                                model = TD3_ActionGNN_old(**kwargs)

                            model.load(
                                filename=f'{load_model_path}model.best')

                        elif "GNN" in algorithm:
                            model = TD3_GNN(**kwargs)
                            algorithm_name = "TD3_GNN"
                            model.load(
                                filename=f'{load_model_path}model.best')

                        else:
                            print("Loading TD3 model")
                            model = TD3(**kwargs)
                            algorithm_name = "TD3"
                            model.load(
                                filename=f'{load_model_path}model.best')

                        if k == 0:
                            actor_model = model.actor
                            model_parameters = filter(
                                lambda p: p.requires_grad, actor_model.parameters())
                            params = sum([np.prod(p.size())
                                          for p in model_parameters])
                            print(
                                f'Actor model has {params} trainable parameters')
                    else:
                        raise ValueError(f'Unknown algorithm {algorithm}')

                # elif algorithm == DecisionTransformer:
                #     model_path = "K=12,embed_dim=128,n_layer=3,max_iters=2000,num_steps_per_iter=20000,batch_size=128,n_head=4,num_eval_episodes=30,scale=1-RR_10_000-231028"
                #     model, state_mean, state_std = load_DT_model(model_path=model_path,
                #                                                  max_ep_len=simulation_length,
                #                                                  env=env,
                #                                                  device=device)

                #     algorithm_name = algorithm.__name__
                #     model.eval()

                else:
                    model = algorithm(env=env,
                                      replay_path=replay_path,
                                      verbose=False)
                    algorithm_name = algorithm.__name__
            except Exception as error:
                print(error)
                print(
                    f'!!!!!!!!!! Error in {algorithm} with replay {replay_path}')
                continue

            rewards = []

            for i in range(simulation_length):
                # print(f' Step {i+1}/{simulation_length} -- {algorithm}')
                ################# Evaluation ##############################

                # elif algorithm == DecisionTransformer:
                #     result_tuple = evaluate_episode_rtg_from_replays(env=env,
                #                                                      model=model,
                #                                                      max_ep_len=simulation_length,
                #                                                      device='cuda',
                #                                                      target_return=0,
                #                                                      mode='normal',
                #                                                      state_mean=state_mean,
                #                                                      state_std=state_std,)
                #     stats, reward = result_tuple
                #     done = True

                if type(algorithm) == str:
                    if any(algo in algorithm for algo in ['ppo', 'a2c', 'ddpg', 'tqc', 'trpo', 'ars', 'rppo']):
                        action, _ = model.predict(state, deterministic=True)
                        obs, reward, done, stats = env.step(action)

                        if i == simulation_length - 2:
                            saved_env = deepcopy(env.get_attr('env')[0])

                        stats = stats[0]
                    elif "SAC" in algorithm or "TD3" in algorithm:
                        action = model.select_action(state,
                                                     return_mapped_action=True)

                        simple_state = state_function_Normal(env=env)
                        gnn_state = state_function_GNN(env=env)

                        ev_indexes = gnn_state['action_mapper']

                        state, reward, done, _, stats = env.step(action)

                        next_simple_state = state_function_Normal(env=env)
                        next_gnn_state = state_function_GNN(env=env)

                        if SAVE_REPLAY_BUFFER:
                            replay_buffers[algorithm].add(state=simple_state,
                                                          action=action,
                                                          ev_action=action[ev_indexes],
                                                          next_state=next_simple_state,
                                                          reward=reward,
                                                          done=done,
                                                          gnn_state=gnn_state,
                                                          gnn_next_state=next_gnn_state)

                else:
                    simple_state = state_function_Normal(env=env)
                    gnn_state = state_function_GNN(env=env)
                    ev_indexes = gnn_state['action_mapper']

                    action = model.get_action(env=env)
                    new_state, reward, done, _, stats = env.step(action)

                    next_simple_state = state_function_Normal(env=env)
                    next_gnn_state = state_function_GNN(env=env)

                    if SAVE_REPLAY_BUFFER:
                        replay_buffers[algorithm].add(state=simple_state,
                                                      action=action,
                                                      ev_action=action[ev_indexes],
                                                      next_state=next_simple_state,
                                                      reward=reward,
                                                      done=done,
                                                      gnn_state=gnn_state,
                                                      gnn_next_state=next_gnn_state)
                ############################################################

                rewards.append(reward)

                if done:

                    results_i = pd.DataFrame({'run': k,
                                              'Algorithm': algorithm_name,
                                              'control_horizon': h,
                                              'discharge_price_factor': config['discharge_price_factor'],
                                              'total_ev_served': stats['total_ev_served'],
                                              'total_profits': stats['total_profits'],
                                              'total_energy_charged': stats['total_energy_charged'],
                                              'total_energy_discharged': stats['total_energy_discharged'],
                                              'average_user_satisfaction': stats['average_user_satisfaction'],
                                              'power_tracker_violation': stats['power_tracker_violation'],
                                              'tracking_error': stats['tracking_error'],
                                              'energy_tracking_error': stats['energy_tracking_error'],
                                              'energy_user_satisfaction': stats['energy_user_satisfaction'],
                                              'total_transformer_overload': stats['total_transformer_overload'],
                                              'battery_degradation': stats['battery_degradation'],
                                              'battery_degradation_calendar': stats['battery_degradation_calendar'],
                                              'battery_degradation_cycling': stats['battery_degradation_cycling'],
                                              'total_reward': sum(rewards),
                                              'time': time.time() - timer,
                                              }, index=[counter])

                    # change name of key to algorithm_name
                    if SAVE_REPLAY_BUFFER:
                        if k == n_test_cycles - 1:
                            replay_buffers[algorithm_name] = replay_buffers.pop(
                                algorithm)

                    if counter == 1:
                        results = results_i
                    else:
                        results = pd.concat([results, results_i])

                    if algorithm in [PPO, A2C, DDPG, SAC, TD3, TQC, TRPO, ARS, RecurrentPPO]:
                        env = saved_env

                    plot_results_dict[algorithm_name] = deepcopy(env)

                    break

    # save the replay buffers to a pickle file
    if SAVE_REPLAY_BUFFER:
        with open(save_path + 'replay_buffers.pkl', 'wb') as f:
            pickle.dump(replay_buffers, f)

    # save the plot_results_dict to a pickle file
    with open(save_path + 'plot_results_dict.pkl', 'wb') as f:
        pickle.dump(plot_results_dict, f)

    # save the results to a csv file
    results.to_csv(save_path + 'data.csv')

    # Group the results by algorithm and print the average and the standard deviation of the statistics
    results_grouped = results.groupby('Algorithm').agg(['mean', 'std'])

    # savethe latex results in a txt file
    # with open(save_path + 'results_grouped.txt', 'w') as f:
    #     f.write(results_grouped.to_latex())

    # results_grouped.to_csv('results_grouped.csv')
    # print(results_grouped[['tracking_error', 'energy_tracking_error']])
    # print(results_grouped[['tracking_error',
    #       'total_transformer_overload', 'time']])
    print(results_grouped[['tracking_error',
                           ]])
    #    'average_user_satisfaction']])
    # input('Press Enter to continue')

    algorithm_names = []
    for algorithm in algorithms:
        # if class has attribute .name, use it
        if hasattr(algorithm, 'algo_name'):
            algorithm_names.append(algorithm.algo_name)
        elif type(algorithm) == str:
            if "GNN" in algorithm:
                algorithm_names.append(algorithm.split(
                    '_')[0] + '_' + algorithm.split('_')[1])
            else:
                algorithm_names.append(algorithm.split('_')[0])
        else:
            algorithm_names.append(algorithm.__name__)

    print(f'Plottting results at {save_path}')

    # plot_total_power(results_path=save_path + 'plot_results_dict.pkl',
    #                  save_path=save_path,
    #                  algorithm_names=algorithm_names)

    # plot_comparable_EV_SoC(results_path=save_path + 'plot_results_dict.pkl',
    #                        save_path=save_path,
    #                        algorithm_names=algorithm_names)

    plot_actual_power_vs_setpoint(results_path=save_path + 'plot_results_dict.pkl',
                                  save_path=save_path,
                                  algorithm_names=algorithm_names)

    # plot_total_power_V2G(results_path=save_path + 'plot_results_dict.pkl',
    #                      save_path=save_path,
    #                      algorithm_names=algorithm_names)

    # plot_comparable_EV_SoC_single(results_path=save_path + 'plot_results_dict.pkl',
    #                               save_path=save_path,
    #                               algorithm_names=algorithm_names)

    # plot_prices(results_path=save_path + 'plot_results_dict.pkl',
    #             save_path=save_path,
    #             algorithm_names=algorithm_names)


if __name__ == "__main__":
    evaluator()
