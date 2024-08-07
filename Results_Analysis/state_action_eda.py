import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import time

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ev2gym.models.ev2gym_env import EV2Gym


class AnalysisReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.ev_action = [{} for i in range(max_size)]
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.gnn_state = [{} for i in range(max_size)]
        self.gnn_next_state = [{} for i in range(max_size)]

        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, ev_action, next_state, reward, done,
            gnn_state, gnn_next_state):

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.ev_action[self.ptr] = ev_action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.gnn_state[self.ptr] = gnn_state
        self.gnn_next_state[self.ptr] = gnn_next_state

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


def action_hist(replay_buffers):

    fig, axs = plt.subplots(3, 4, figsize=(10, 6))

    for i, algo in enumerate(replay_buffers.keys()):
        print(f"Algorithm: {algo}")

        actions = replay_buffers[algo].ev_action[:replay_buffers[algo].size]
        # actions = replay_buffers[algo].action[:replay_buffers[algo].size]

        if i == 0:
            print(f"Number of state-action pairs: {len(actions)}")

        # flatten the list of actions
        actions = [item for sublist in actions for item in sublist]

        # turn negative actions to 0
        actions = [0 if action < 0 else action for action in actions]

        plt.subplot(3, 4, i+1)

        plt.hist(actions,
                 bins=np.linspace(0, 1, 100),
                 )

        plt.title(f"Algorithm: {algo}")

    plt.show()


def plot_TSNE_state_action(replay_buffers):

    fig, axs = plt.subplots(3, 4, figsize=(10, 6))

    for i, algo in enumerate(replay_buffers.keys()):
        print(f"Algorithm: {algo}")

        x = replay_buffers[algo].state[:replay_buffers[algo].size]
        y = replay_buffers[algo].action[:replay_buffers[algo].size]
        # y = [sum(action) for action in y]

        # make Dataframe
        df = pd.DataFrame(y)

        df_columns = df.columns

        # df['action'] = y

        df['pst'] = replay_buffers[algo].state[:replay_buffers[algo].size][:, 1]

        ev_num = replay_buffers[algo].ev_action[:replay_buffers[algo].size]

        # find the length of each sublist
        df['ev_num'] = [len(i) for i in ev_num]

        # print max y

        # discretize actions into 3 bins
        df['ev_num'] = pd.cut(df['ev_num'],
                              bins=10,
                              labels=False)

        data_subset = df[df_columns].values

        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=2000)
        tsne_results = tsne.fit_transform(data_subset)

        # print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

        # print(df.head())

        df['tsne-2d-one'] = tsne_results[:, 0]
        df['tsne-2d-two'] = tsne_results[:, 1]

        plt.subplot(3, 4, i+1)
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="ev_num",
            # palette=sns.color_palette("hls", 10),
            data=df,
            legend="full",
            alpha=0.3
        )

    plt.show()


def plot_hist_vs_ev_num(replay_buffers):

    fig, axs = plt.subplots(3, 4, figsize=(10, 6))

    for i, algo in enumerate(replay_buffers.keys()):
        print(f"Algorithm: {algo}")

        ev_num = replay_buffers[algo].ev_action[:replay_buffers[algo].size]

        # find the length of each sublist
        ev_num = [len(i) for i in ev_num]

        # make 5 bins for the ev_num
        ev_num = pd.cut(ev_num,
                        bins=np.linspace(-1, max(ev_num), 5),
                        labels=False)
        # print(ev_num)

        plt.subplot(3, 4, i+1)

        # make len(ev_num) dictioanry of lists
        ev_num_dict = {i: [] for i in range(max(ev_num)+1)}

        # make lists for each ev number
        action_list = []

        # iterate the replay buffer.ev_action
        for j in range(len(replay_buffers[algo].ev_action[:replay_buffers[algo].size])):

            ev_num_dict[ev_num[j]].extend(
                replay_buffers[algo].ev_action[:replay_buffers[algo].size][j])

        for key in ev_num_dict.keys():
            # if actions are negative turn them to 0
            ev_num_dict[key] = [0 if action <
                                0 else action for action in ev_num_dict[key]]

            plt.hist(ev_num_dict[key],
                     bins=np.linspace(0, 1, 50),
                     alpha=0.3,
                     label=f"EV number: {key}"
                     )

        plt.legend()
        # plt.show()
    # save the figure
    plt.savefig("./Results_Analysis/hist_vs_ev_num.png")

    plt.show()


def plot_violin_actions(replay_buffers):

    fig, axs = plt.subplots(3, 4, figsize=(10, 6))

    # initialize dataframe with columns algo, ev_num, action
    df = pd.DataFrame(columns=['algo', 'ev_num', 'action'])

    for i, algo in enumerate(replay_buffers.keys()):
        print(f"Algorithm: {algo}")

        ev_num = replay_buffers[algo].ev_action[:replay_buffers[algo].size]

        # find the length of each sublist
        ev_num = [len(i) for i in ev_num]

        # make 5 bins for the ev_num
        ev_num = pd.cut(ev_num,
                        bins=np.linspace(-1, max(ev_num), 4),
                        labels=False)
        # print(ev_num)

        plt.subplot(3, 4, i+1)

        # make len(ev_num) dictioanry of lists
        ev_num_dict = {i: [] for i in range(max(ev_num)+1)}

        # make lists for each ev number
        action_list = []

        # iterate the replay buffer.ev_action
        for j in range(len(replay_buffers[algo].ev_action[:replay_buffers[algo].size])):

            ev_num_dict[ev_num[j]].extend(
                replay_buffers[algo].ev_action[:replay_buffers[algo].size][j])

            temp_df = pd.DataFrame({'algo': [algo]*len(replay_buffers[algo].ev_action[:replay_buffers[algo].size][j]),
                                    'ev_num': [ev_num[j]]*len(replay_buffers[algo].ev_action[:replay_buffers[algo].size][j]),
                                    'action': replay_buffers[algo].ev_action[:replay_buffers[algo].size][j]})

            df = pd.concat([df, temp_df])

        for key in ev_num_dict.keys():
            # if actions are negative turn them to 0
            ev_num_dict[key] = [0 if action <
                                0 else action for action in ev_num_dict[key]]

    #         sns.violinplot(data=ev_num_dict[key],
    #                     #    hue=ev_num,
    #                     inner=None,

    #                     linewidth=1.5,
    #                     alpha=0.2,
    #                     density_norm = 'area',

    #                     # label=f"EV number: {key}"
    #                     )
    #     if i == 9:
    #         plt.legend()
    #     # plt.show()
    # # save the figure
    # plt.savefig("./Results_Analysis/violin_actions.png")

    # save df
    df.to_csv("./Results_Analysis/violin_actions.csv")


def violin_plot(df):

    # print uniqe algo
    print(df.algo.unique())
    # rename algo

    df['algo'] = df['algo'].replace({"<class 'ev2gym.baselines.heuristics.ChargeAsFastAsPossible'>": 'AFAP',
                                     "ChargeAsFastAsPossible": 'AFAP',
                                     "RoundRobin": 'RR',
                                     "<class 'ev2gym.baselines.heuristics.RoundRobin_GF'>": 'RR_GF',
                                     "<class 'ev2gym.baselines.heuristics.RoundRobin'>": 'RR',
                                    "<class 'ev2gym.baselines.gurobi_models.tracking_error.PowerTrackingErrorrMin'>": 'Optimal',
                                    "PowerTrackingErrorrMin": 'Optimal'}
                                    )

    # if algo_name is RR drop these rows
    df = df[df.algo != 'RR_GF']

    # change algo if string is in algo
    df['algo'] = df['algo'].apply(
        lambda x: 'SAC \n GNN-FX' if 'SAC_GNN' in x else x)
    df['algo'] = df['algo'].apply(
        lambda x: 'SAC \n EV-GNN' if 'SAC_ActionGNN' in x else x)
    df['algo'] = df['algo'].apply(lambda x: 'SAC' if 'SAC_run' in x else x)

    df['algo'] = df['algo'].apply(
        lambda x: 'TD3 \n GNN-FX' if 'TD3_GNN' in x else x)
    df['algo'] = df['algo'].apply(
        lambda x: 'TD3 \n EV-GNN' if 'TD3_ActionGNN' in x else x)
    df['algo'] = df['algo'].apply(lambda x: 'TD3' if 'TD3_run' in x else x)

    # print column names
    print(df.columns)

    # if action is negative turn it to 0
    df['action'] = df['action'].apply(lambda x: 0 if x < 0 else x)

    colors = ["#008585", "#fbf2c4", "#c7522a"]
    ev_number_labels = ["  0-33 %", " 34-66 %", "67-100 %"]

    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['font.family'] = ['serif']
    
    for i in range(3):
        sns.violinplot(data=df[df['ev_num'] == i],
                       x='algo',
                       y='action',
                       inner=None,
                       order=['AFAP', 'RR',
                              'SAC', 'TD3',
                              'SAC \n GNN-FX', 'TD3 \n GNN-FX',
                              'SAC \n EV-GNN', 'TD3 \n EV-GNN',
                              'Optimal'],
                       
                       cut=0,
                       split=True,
                       color=colors[i],
                       alpha=0.7,
                       density_norm='count',
                       saturation=0.75,
                       common_norm=True,
                       )

    # Compose a custom legend
    custom_lines = [
        Line2D([0], [0], color=colors[i], lw=3, alpha=0.6)
        for i in range(len(ev_number_labels))
    ]
    
    ax.legend(custom_lines,
              ev_number_labels,
              loc='center left',
              title='Parked EVs')

    # add horizonal gridlines
    ax.set_axisbelow(True)
    ax.yaxis.grid(True,
                    linestyle='--',
                    alpha=0.7)
    
    # drop x axis label
    plt.xlabel('')
    plt.ylabel('Action', fontsize=14, font='serif')
    #change xticks size
    plt.xticks(fontsize=14,font='serif',rotation=45)
    plt.yticks(fontsize=13,font='serif')
    
    plt.ylim(0, 1)
    
    #remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
    
    # save the figure
    plt.savefig("./Results_Analysis/violin_actions.png")
    
    plt.show()

if __name__ == "__main__":

    # Load the replay buffer pickle file from ./results

    # results_path = "./results/eval_25cs_3tr_PublicPST_8_algos_5_exp_2024_07_11_912438/"    
    # results_path = "./results/eval_100cs_7tr_PublicPST_100_10_algos_5_exp_2024_07_11_353226/"
    # results_path = "./results/eval_100cs_7tr_PublicPST_100_9_algos_30_exp_2024_07_12_312179/"
    
    # results_path = "./results/eval_500cs_35tr_PublicPST_500_9_algos_2_exp_2024_07_17_916277/"

    results_path = "./results/eval_25cs_3tr_PublicPST_9_algos_30_exp_2024_07_23_602483/"
    # Load the replay buffer
    replay_buffers = pickle.load(
        open(f"{results_path}replay_buffers.pkl", "rb"))

    # Plot the action histograms
    # action_hist(replay_buffers)

    # plot TSNE of the state-action pairs
    # plot_TSNE_state_action(replay_buffers)

    # plot the distribution of actions when ev number is fixed
    # plot_hist_vs_ev_num(replay_buffers)

    # plot violinsplots of action distributions
    plot_violin_actions(replay_buffers)

    # load df
    df = pd.read_csv("./Results_Analysis/violin_actions.csv")
    # plot
    violin_plot(df)
