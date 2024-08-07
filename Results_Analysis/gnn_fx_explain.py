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

import networkx as nx
import matplotlib.pyplot as plt
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from PIL import Image
from sklearn.manifold import TSNE
import torch.nn.functional as F

import seaborn as sns

import shap

from ev2gym.models.ev2gym_env import EV2Gym

from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, SimpleReward
from ev2gym.rl_agent.reward import profit_maximization, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads

# GNN-based models evaluations
from TD3.TD3_GNN import TD3_GNN
# from TD3.TD3_ActionGNN import TD3_ActionGNN
from TD3.old_TD3_ActionGNN import TD3_ActionGNN
from TD3.TD3 import TD3
from SAC.sac import SAC
from SAC.actionSAC import SAC_ActionGNN
from GNN.state import PublicPST_GNN, V2G_ProfitMax_with_Loads_GNN

from state_action_eda import AnalysisReplayBuffer

config_file = "./config_files/PublicPST.yaml"  # 25
# config_file = "./config_files/PublicPST_100.yaml"
# config_file = "./config_files/PublicPST_500.yaml"
# config_file = "./config_files/PublicPST_1000.yaml"

# config_file = "./config_files/GF_PST_25.yaml"
# config_file = "./config_files/GF_PST_100.yaml"
# config_file = "./config_files/GF_PST_500.yaml"

# config_file = "/config_files/V2G_ProfixMaxWithLoads_25.yaml"
# config_file = "/config_files/V2G_ProfixMaxWithLoads_100.yaml"
# config_file = "/config_files/V2G_ProfixMaxWithLoads_500.yaml"

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


algorithms = [
    # PST 25CS
    # "SAC_run_0_25377-546286",
    # "SAC_GNN_run_0_6651-167314",
    # "SAC_ActionGNN_run_0_39335-597033",
    "TD3_ActionGNN_run_3_99695-870291",
    # "TD3_GNN_run_2_25852-556176",
    # "TD3_run_0_47448-558478",

    # PST 100cs
    # "SAC_ActionGNN_run_2_57591-144641",
    # "SAC_GNN_run_0_16099-876950",
    # "SAC_run_2_25877-194576",
    # "TD3_run_4_94486-719210",
    # "TD3_GNN_run_4_14896-733342",
    # "TD3_ActionGNN_run_2_31792-403621",

]


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

    return replay_path


replay_path_env = generate_replay("GNN_fx_explain")

for algorithm in algorithms:

    if "GNN" in algorithm:
        state_function = state_function_GNN
    else:
        state_function = state_function_Normal

    env = EV2Gym(config_file=config_file,
                 load_from_replay_path=replay_path_env,
                 state_function=state_function,
                 reward_function=reward_function,
                 )

    state, _ = env.reset()

    if "SAC" in algorithm:
        load_model_path = f'./eval_models/{algorithm}/'
        # Load kwargs.yaml as a dictionary
        with open(f'{load_model_path}kwargs.yaml') as file:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)

        if hasattr(state_function, 'node_sizes'):
            fx_node_sizes = state_function.node_sizes

        if "ActionGNN" in algorithm:
            model = SAC_ActionGNN(action_space=env.action_space,
                                  fx_node_sizes=fx_node_sizes,
                                  args=kwargs,)

            algorithm_name = "SAC_ActionGNN"

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

    elif "TD3" in algorithm:
        load_model_path = f'./eval_models/{algorithm}/'
        # Load kwargs.yaml as a dictionary
        with open(f'{load_model_path}kwargs.yaml') as file:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)

        if "ActionGNN" in algorithm:
            print("Loading TD3_ActionGNN")
            model = TD3_ActionGNN(**kwargs)

            algorithm_name = "TD3_ActionGNN"
            model.load(filename=f'{load_model_path}model.best')

        elif "GNN" in algorithm:
            model = TD3_GNN(**kwargs)
            algorithm_name = "TD3_GNN"
            model.load(filename=f'{load_model_path}model.best')

        else:
            print("Loading TD3 model")
            model = TD3(**kwargs)
            algorithm_name = "TD3"
            model.load(filename=f'{load_model_path}model.best')

    # action = model.select_action(state,
    #                              return_mapped_action=True)

    # # simple_state = state_function_Normal(env=env)
    # # gnn_state = state_function_GNN(env=env)

    # # ev_indexes = gnn_state['action_mapper']

    # state, reward, done, _, stats = env.step(action)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # PST 25 EVSE
    replay_path = "./results/eval_25cs_3tr_PublicPST_1_algos_1_exp_2024_08_01_183555/replay_buffers.pkl"

    # PST 100 EVSE
    # replay_path = f"./results/eval_100cs_7tr_PublicPST_100_9_algos_30_exp_2024_07_12_312179/replay_buffers.pkl"

    data = pickle.load(open(replay_path, 'rb'))
    print(data.keys())

    # algo_name = "TD3_run_0_47448-558478"
    algo_name = "TD3_ActionGNN"

    print(data[algo_name].state.shape)
    size = data[algo_name].size
    print(f' Size: {size}')

    if "GNN" in algorithm:

        X_test = data[algo_name].gnn_state[:size]
        print(f'X_test shape: {X_test[15]}')

        gnn = model.actor
        gnn_model = model
    else:
        X_test = data[algo_name].state[:size]
        # turn to tensor
        X_test = torch.tensor(X_test,
                              dtype=torch.float32,
                              device=device)

        if "TD3" in algorithm:
            mlp = model.actor
        else:
            raise ValueError(f'Unknown algorithm {algorithm}')

    if "GNN" in algorithm:
        # =================================================================
        # Vizualizing the GCN outputs
        print(f' \n\n GNN EXPLAIN \n\n')
        # networkx build graph from edge index

        sample_numbner = 100  # 90

        edge_index = X_test[sample_numbner].edge_index
        # # for i in range(edge_index.shape[1]):
        G = nx.from_edgelist(edge_index.T)

        outputs = gnn.explain_GCN(X_test[sample_numbner])
        # actions = gnn.select_action(X_test[sample_numbner])

        figure = plt.figure(figsize=(10, 3))
        plt.rcParams.update({'font.size': 12})
        plt.rcParams['font.family'] = ['serif']

        for i, embeddings in enumerate(outputs):
            ax = figure.add_subplot(1, len(outputs), i+1)
            print(f'Output {i} shape: {embeddings.shape}')

            # put the original full graph
            if i == 0:
                pos = nx.spring_layout(G,
                                       iterations=100,
                                       # threshold=1e-5,
                                       center=[0, 100],
                                       seed=30)

            if i == 0:
                title = 'Embedded \n Features ($F^0=32$)'
            elif i == 1:
                title = '$1^{st}$ GCN Layer \n Output ($F^1=64$)'
            elif i == 2:
                title = '2$^{nd}$ GCN Layer \n Output ($F^2=32$)'
            elif i == 3:
                title = 'Actor Output\n ($F^L=1$)'

            embeddings = embeddings.mean(axis=1)
            # embeddings = embeddings.max(axis=1)
            nx.draw(G, pos,
                    ax=ax,
                    with_labels=True,
                    node_color=embeddings,
                    cmap=plt.cm.viridis,
                    # node_color='skyblue',
                    node_size=210,
                    font_size=10,
                    font_color='slateblue',
                    edgecolors='gray',
                    # title=title,
                    # xlabel=title,
                    )
            ax.set_title(title)
            # ax.set_xlabel('$F_{i}$')

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            # Create a divider for the existing axes instance
            divider = make_axes_locatable(ax)
            # Append axes to the right of the current axes, with 5% width of the figure
            cax = divider.append_axes("right", size="5%", pad=0.05)

            # Create a colorbar with the custom Axes
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(
                vmin=embeddings.min(), vmax=embeddings.max()))
            sm._A = []
            cbar = plt.colorbar(sm, cax=cax)

            # Change the font size of the colorbar labels
            cbar.ax.tick_params(labelsize=11)  # Adjust the fontsize as needed

        # make left margin wider
        # plt.subplots_adjust(left=0, right=0.95, top=1, bottom=0.2)
        plt.tight_layout()
        plt.show()
        

        load_model_path = f'./eval_models/TD3_run_0_47448-558478/'
        with open(f'{load_model_path}kwargs.yaml') as file:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)

        model = TD3(**kwargs)
        algorithm_name = "TD3"
        model.load(filename=f'{load_model_path}model.best')

        state = data[algo_name].state[sample_numbner]
        state = torch.tensor(state,
                             dtype=torch.float32,
                             device=device)

        print(f'GNN action: {gnn_model.select_action(X_test[sample_numbner])}')
        print(F.relu(model.actor(state)))

        gnn_actions = gnn_model.select_action(X_test[sample_numbner])
        # make negative values 0 of my numpy array
        gnn_actions = np.maximum(0, gnn_actions)

        mlp_actions = F.relu(model.actor(state)).cpu().detach().numpy()

        ev_mapper = X_test[sample_numbner].action_mapper

        # make df with columns, action, evse, algorithm
        df = pd.DataFrame(columns=['action', 'evse', 'algorithm'])
        for i in range(len(gnn_actions)):
            df = pd.concat([df, pd.DataFrame(
                {'action': [mlp_actions[i]], 'evse': [i], 'algorithm': ['TD3']})])
            df = pd.concat([df, pd.DataFrame(
                {'action': [gnn_actions[i]], 'evse': [i], 'algorithm': ['TD3 EV-GNN']})])

        df.reset_index(drop=True, inplace=True)

        # print(df)

        fig = plt.figure(figsize=(10, 3))
        plt.rcParams.update({'font.size': 12})
        plt.rcParams['font.family'] = ['serif']

        ax = sns.barplot(data=df,
                         x='evse',
                         y='action',
                         hue="algorithm",
                         palette=['skyblue', '#E74C3C'],
                         alpha=0.8,
                         edgecolor='k',
                         saturation=1,
                         zorder=5)

        plt.ylabel('Action', fontsize=15)
        plt.xlabel('Charging Station (EVSE)', fontsize=15)
        plt.xticks(fontsize=13)
        plt.ylim(0, 1)

        # put legend over the plot
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=2, fontsize=13,
                   framealpha=1)

        # show grid for vertical lines
        plt.grid(axis='y', linestyle='--', linewidth=0.5)

        # make a vector of size 25 where the element i = 1 if the evse is in the ev_mapper
        ev_mapper = [1 if i in ev_mapper else 0 for i in range(25)]
        second_color = "#5acc61"
        ax = sns.barplot(x=range(25),
                         y=ev_mapper,
                         color=second_color,
                         alpha=0.8,
                         # saturation=1,
                         width=1,
                         zorder=0
                         )

        # plot vertical lines to separate the evses, between the evses
        for i in range(24):
            plt.axvline(x=i+0.5, color='black', linestyle='--',
                        linewidth=0.8, zorder=1)

        from PIL import Image
        image_path = './Results_Analysis/icon/charging-station.png'

        img = mpimg.imread(image_path)
        with Image.open(image_path) as img:
            resized_img = img.resize((30, 30))
            
        image_path_2 = './Results_Analysis/icon/car.png'
        
        img_2 = mpimg.imread(image_path_2)
        with Image.open(image_path_2) as img_2:
            resized_img_2 = img_2.resize((30, 30))
    
        # add an icon over the evses that are in the ev_mapper
        print(f'EV Mapper: {ev_mapper}')
        counter = 0
        for i in range(25):
            if ev_mapper[i] == 1:
                # xo is over the plot
                fig.figimage(resized_img,
                             xo= 31*i+130,
                             yo=270,                            
                             alpha=1,
                             zorder=1)
            # else:
            #     fig.figimage(resized_img_2,
            #                  xo= 31*i+130,
            #                  yo=270,                            
            #                  alpha=1,
            #                  zorder=1)
            counter += 1

        # add a second axis on the right side
        plt.twinx()

        second_color = "#6d9468"
        # plot the ev_mapper as barplot
        # ax2 =  sns.barplot(x=range(25),
        #             y=ev_mapper,
        #             color=second_color,
        #             alpha=0.8,
        #             saturation=1,
        #             # zorder=0
        #             )

        # move barplot to the background of the plot
        # plt.gca().set_zorder(1)
        # ax2.set_zorder(5)
        # plt.gca().patch.set_visible(False)

        # ax.set_zorder(10)

        # plt.ylabel('EV Parked at EVSE $i$', color=second_color,
        #            fontsize=15)

        # change the color of the y-axis ticks and labels
        plt.ylim(0, 1)
        # plt.yticks([0, 1], ['No', 'Yes'], color=second_color, fontsize=15)
        # dont show the y-axis ticks
        plt.yticks([])

        plt.show()

        continue

    # Apply t-SNE

    # figure = plt.figure(figsize=(10, 10))

    # for i, embeddings in enumerate([o1, o2, o3, o4, o5, o6]):
    #     print(f'Output {i} shape: {embeddings.shape}')
    #     if embeddings.shape[1] == 1:
    #         n_components = 1
    #     else:
    #         n_components = 2

    #     tsne = TSNE(n_components=n_components)
    #     embeddings_2d = tsne.fit_transform(embeddings)

    #     # Plot the 2D embeddings in subplots
    #     ax = figure.add_subplot(2, 4, i+1)

    #     if n_components == 1:
    #         ax.scatter(range(embeddings_2d.shape[0]),
    #                    embeddings_2d[:, 0],
    #                    c='skyblue',
    #                    edgecolors='k')
    #     else:
    #         ax.scatter(embeddings_2d[:, 0],
    #                 embeddings_2d[:, 1],
    #                 c='skyblue',
    #                 edgecolors='k')
    #     ax.set_title(f'Output {i} Visualized using t-SNE')
    #     ax.set_xlabel('Dimension 1')
    #     ax.set_ylabel('Dimension 2')
    # tsne = TSNE(n_components=2)
    # embeddings_2d = tsne.fit_transform(o2)

    # # Plot the 2D embeddings in subplots

    # plt.scatter(embeddings_2d[:, 0], embeddings_2d[:,
    #             1], c='skyblue',
    #             edgecolors='k')
    # plt.title('Node Embeddings Visualized using t-SNE')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.show()

    # exit(0)

    # =================================================================
    # explainer = shap.DeepExplainer(mlp, X_test)
    # shap_values = explainer.shap_values(X_test, check_additivity=False)

    # #pickle the shap values
    # with open('./Results_Analysis/shap_values.pkl', 'wb') as f:
    #     pickle.dump(shap_values, f)
    # #pickle the explainer
    # with open('./Results_Analysis/explainer.pkl', 'wb') as f:
    #     pickle.dump(explainer, f)

    # load the shap values
    with open('./Results_Analysis/shap_values.pkl', 'rb') as f:
        shap_values = pickle.load(f)
    with open('./Results_Analysis/explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)

    # print(f'shap_values shape: {shap_values.shape}')
    # Plot feature importance
    # shap.summary_plot(shap_values, X_test)
    # Plot summary of SHAP values to visualize feature importance
    print(X_test.shape[1])

    import shap

    # Assuming you have feature names
    feature_names = [f'Feature {i}' for i in range(78)]

    X_test_np = X_test.cpu().numpy()

    # Summary Plot for Each Output
    # Loop through each output
    # for output_idx in range(25):
    #     print(f'Summary plot for Output {output_idx + 1}')
    #     # SHAP values for the current output
    #     shap_values_output = shap_values[:, :, output_idx]

    #     # Plot summary for the current output
    #     shap.summary_plot(shap_values_output, X_test_np, feature_names=feature_names)

    # Waterfall Plot for Each Output
    # for output_idx in range(25):
    #     print(f'Waterfall plot for Output {output_idx + 1}')
    #     # SHAP values for the current output
    #     shap_values_output = shap_values[:, :, output_idx]

    #     # Plot waterfall for a single instance, say the first instance
    #     shap.waterfall_plot(shap.Explanation(values=shap_values_output[0], base_values=explainer.expected_value[output_idx], data=X_test_np[0], feature_names=feature_names))

    # Decision Plot for Each Output
    for output_idx in range(25):
        print(f'Decision plot for Output {output_idx + 1}')
        # SHAP values for the current output
        shap_values_output = shap_values[:, :, output_idx]

        # Plot decision plot for the current output
        shap.decision_plot(
            explainer.expected_value[output_idx], shap_values_output, X_test_np, feature_names=feature_names)

    # Force Plot for Each Output
    # for output_idx in range(25):
    #     print(f'Force plot for Output {output_idx + 1}')
    #     # SHAP values for the current output
    #     shap_values_output = shap_values[:, :, output_idx]

    #     # Plot force for a single instance, say the first instance
    #     shap.force_plot(explainer.expected_value[output_idx], shap_values_output[0], X_test_np[0], feature_names=feature_names)
    #     plt.show()

    # Dependence Plot for Each Output
    # for output_idx in range(25):
    #     print(f'Dependence plot for Output {output_idx + 1}')
    #     # SHAP values for the current output
    #     shap_values_output = shap_values[:, :, output_idx]

    #     # Plot dependence for a specific feature, say feature index 0
    #     shap.dependence_plot(0, shap_values_output, X_test_np, feature_names=feature_names)

    # Bar Plot for Each Output
    # for output_idx in range(25):
    #     print(f'Bar plot for Output {output_idx + 1}')
    #     # SHAP values for the current output
    #     shap_values_output = shap_values[:, :, output_idx]

    #     # Plot bar plot for the current output
    #     shap.summary_plot(shap_values_output, X_test_np, feature_names=feature_names, plot_type="bar")
