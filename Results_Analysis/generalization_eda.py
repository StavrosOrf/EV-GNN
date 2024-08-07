import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import time
import matplotlib.ticker as ticker

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.patches

# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

from ev2gym.models.ev2gym_env import EV2Gym


class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
                isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(
            n, m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(
            r+1, r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def plot_optimality_gap_in_generalization():
    config_files = [
        "./results/eval_500cs_35tr_PublicPST_500_9_algos_100_exp_2024_07_28_413702/",
        "./results/eval_500cs_35tr_PublicPST_500_g3_9_algos_100_exp_2024_07_28_867729/",
        "./results/eval_500cs_35tr_PublicPST_500_g1_9_algos_100_exp_2024_07_28_235045/",
        "./results/eval_500cs_35tr_PublicPST_500_g2_9_algos_100_exp_2024_07_28_316433/",
    ]

    config_files = [
        "./results/eval_100cs_7tr_PublicPST_100_9_algos_100_exp_2024_07_28_640338/",
        "./results/eval_100cs_7tr_PublicPST_100_g3_9_algos_100_exp_2024_07_28_787462/",
        "./results/eval_100cs_7tr_PublicPST_100_g1_9_algos_100_exp_2024_07_28_488336/",
        "./results/eval_100cs_7tr_PublicPST_100_g2_9_algos_100_exp_2024_07_28_430526/",
    ]

    columns_to_keep = [
        'run',
        'Algorithm',
        'total_reward',
        'energy_tracking_error',
        'time',
    ]

    figure = plt.figure(figsize=(12, 3))

    all_data = pd.DataFrame()

    for index, config_file in enumerate(config_files):
        with open(config_file + "data.csv", "r") as f:
            # read as pandas dataframe
            data = pd.read_csv(f)

        data = data[columns_to_keep]
        # print(data.columns)
        # print(data.Algorithm.unique())
        # print(data.head(20))

        # for every "run" find the optimality gap for each algorithm
        # Optimality gap = (Algorithm.total_reward - PowerTrackingErrorrMin.total_reward) / PowerTrackingErrorrMin.total_reward

        # find the PowerTrackingErrorrMin.total_reward
        PowerTrackingErrorrMin = data[data.Algorithm ==
                                      "PowerTrackingErrorrMin"]
        # print(PowerTrackingErrorrMin.head(20))

        # find the mean and std of the optimality gap for each algorithm

        data = data[data.Algorithm != "PowerTrackingErrorrMin"]
        for i, row in data.iterrows():
            run = row.run
            optimal = PowerTrackingErrorrMin.iloc[run].total_reward
            reward = row.total_reward
            data.at[i, 'G'] = ((reward - optimal) / optimal) * 100

            optimal_energy_tracking_error = PowerTrackingErrorrMin.iloc[run].energy_tracking_error
            energy_tracking_error = row.energy_tracking_error
            data.at[i, 'G_E'] = (
                (energy_tracking_error - optimal_energy_tracking_error) / optimal_energy_tracking_error) * 100

            if index == 0:
                case_name = "Original"
            elif index == 1:
                case_name = "Small"
            elif index == 2:
                case_name = "Medium"
            elif index == 3:
                case_name = "Extreme"
            data.at[i, 'case'] = case_name

        data['Algorithm'] = data['Algorithm'].replace({"<class 'ev2gym.baselines.heuristics.ChargeAsFastAsPossible'>": 'AFAP',
                                                       "ChargeAsFastAsPossible": 'AFAP',
                                                       "RoundRobin": 'RR',
                                                       "<class 'ev2gym.baselines.heuristics.RoundRobin_GF'>": 'RR_GF',
                                                       "<class 'ev2gym.baselines.heuristics.RoundRobin'>": 'RR',
                                                       "<class 'ev2gym.baselines.gurobi_models.tracking_error.PowerTrackingErrorrMin'>": 'Optimal',
                                                       "PowerTrackingErrorrMin": 'Optimal'}
                                                      )

        # change Algorithm if string is in Algorithm
        data['Algorithm'] = data['Algorithm'].apply(
            lambda x: '  SAC  \nGNN-FX' if 'SAC_GNN' in x else x)
        data['Algorithm'] = data['Algorithm'].apply(
            lambda x: '  SAC  \nEV-GNN' if 'SAC_ActionGNN' in x else x)
        data['Algorithm'] = data['Algorithm'].apply(
            lambda x: 'SAC' if 'SAC_run' in x else x)

        data['Algorithm'] = data['Algorithm'].apply(
            lambda x: '  TD3  \nGNN-FX' if 'TD3_GNN' in x else x)
        data['Algorithm'] = data['Algorithm'].apply(
            lambda x: '  TD3  \nEV-GNN' if 'TD3_ActionGNN' in x else x)
        data['Algorithm'] = data['Algorithm'].apply(
            lambda x: 'TD3' if 'TD3_run' in x else x)

        all_data = pd.concat([all_data, data])

        # plt.subplot(1, len(config_files), index+1)

        # sns.boxplot(x="Algorithm",
        #             y="G_E",
        #             data=data)
        # plt.xticks(rotation=45)

        # plt.ylim(-0, 700)
        # if i == 0:
        #     plt.ylabel('Optimality Gap (%)')
        # else:
        #     plt.ylabel('')

        # print(data.head(20))

    # sns.boxplot(x="case",
    #             y="G_E",
    #             hue="Algorithm",
    #             data=all_data)

    # sns.violinplot(x="case",
    #             y="G_E",
    #             hue="Algorithm",
    #             data=all_data,
    #             inner=None)
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['font.family'] = ['serif']

    # ax = sns.boxplot(x="Algorithm",
    #             y="G",
    #             #remove outliers
    #             showfliers=False,
    #             hue="case",
    #             order=['AFAP', 'RR',
    #                    'SAC', 'TD3',
    #                           '  SAC  \nGNN-FX', '  TD3  \nGNN-FX',
    #                           '  SAC  \nEV-GNN', '  TD3  \nEV-GNN'],
    #             data=all_data)

    # use violin plot instead
    
    ax = sns.violinplot(x="Algorithm",
                        y="G",
                        hue="case",
                        order=['AFAP', 'RR',
                               'SAC', 'TD3',
                               '  SAC  \nGNN-FX', '  TD3  \nGNN-FX',
                               '  SAC  \nEV-GNN', '  TD3  \nEV-GNN'],
                        data=all_data,
                        # inner=None,
                        density_norm='width',
                        common_norm=False,
                        native_scale=True,
                        inner_kws=dict(box_width=2, whis_width=1, color=".5")
                        )
    

    # add grid
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # add vetical lines to split the cases
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=1.5, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=2.5, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=3.5, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=4.5, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=5.5, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=6.5, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=7.5, color='black', linestyle='--', alpha=0.5)

    # plt.legend(loc='lower left',
    #            title="Case Variation:",
    #            ncol=4,
    #            frameon=True,
    #            fontsize=12)
    # remove the legend
    ax.legend().remove()

    sns.despine()

    plt.ylabel('Optimality Gap (%)', fontsize=13)

    plt.xlabel('')

    plt.xticks(fontsize=12)
    # plt.ylim(-0, 4500)
    # log scale

    ticks_list = np.arange(0, 10, 1)
    ticks_list = np.concatenate((ticks_list, np.arange(10, 100, 10)))
    ticks_list = np.concatenate((ticks_list, np.arange(100, 1000, 100)))
    ticks_list = np.concatenate((ticks_list, np.arange(1000, 10000, 1000)))
    ticks_list = np.concatenate((ticks_list, np.arange(10000, 100000, 1000)))

    major_ticks = [0, 10, 100, 1000, 10_000]

    # set major ticks locator
    ax.yaxis.set_major_locator(ticker.FixedLocator(major_ticks))

    # set minor ticks formatter
    ax.yaxis.set_minor_locator(ticker.LogLocator(
        base=10.0, subs=(1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001)))

    # add grid for the major ticks
    plt.grid(axis='y', which='major', linestyle='-', alpha=0.9)

    # add grid for the minor ticks
    plt.grid(axis='y', which='minor', linestyle='--', alpha=0.5)

    ax.set_axisbelow(True)
    plt.ylim(1, 15_000)
    plt.yscale('log',
               base=10,
               )
    plt.title('100 EVSE', fontsize=14)

    # yticks = np.arange(0, 1000, 100)
    # plt.yticks(yticks, [f'{y}' for y in yticks], fontsize=12)

    plt.tight_layout()
    plt.savefig("./Results_Analysis/generalization_eda.png")
    plt.show()


def plot_scenario_distributions():

    ev_profile_files = [
        "./results/eval_100cs_7tr_PublicPST_100_9_algos_200_exp_2024_07_28_971759/ev_profiles.pkl",
        "./results/eval_100cs_7tr_PublicPST_100_g3_9_algos_200_exp_2024_07_28_276041/ev_profiles.pkl",
        "./results/eval_100cs_7tr_PublicPST_100_g1_9_algos_200_exp_2024_07_28_572520/ev_profiles.pkl",
        "./results/eval_100cs_7tr_PublicPST_100_g2_9_algos_200_exp_2024_07_28_293670/ev_profiles.pkl",]

    # Simulation Starting Time
    # Hour and minute do not change after the environment has been reset
    hour = 5  # Simulation starting hour (24 hour format)
    minute = 0  # Simulation starting minute (0-59)

    timescale = 15  # in minutes per step
    simulation_length = 112  # 90 # in steps per simulation

    all_data = pd.DataFrame()

    for index, ev_profile_file in enumerate(ev_profile_files):

        with open(ev_profile_file, "rb") as f:
            ev_profiles = pickle.load(f)

        if index == 0:
            case_name = "Original"
        elif index == 1:
            case_name = "Small"
        elif index == 2:
            case_name = "Medium"
        elif index == 3:
            case_name = "Extreme"

        EVs = {"arrival_time": [],
               "departure_time": [],
               "SoC_at_arrival": [],
               "battery_capacity": [],
               "charging_power": [],
               "time_of_stay": [],
               "case": case_name
               }

        for EV in ev_profiles[0]:
            # print(EV)

            arrival_time = EV.time_of_arrival * timescale + hour * 60 + minute
            departure_time = EV.time_of_departure * timescale + hour * 60 + minute

            if arrival_time > 1440:
                arrival_time = arrival_time - 1440
            if departure_time > 1440:
                departure_time = departure_time - 1440

            SoC_at_arrival = (EV.battery_capacity_at_arrival /
                              EV.battery_capacity) * 100
            battery_capacity = EV.battery_capacity
            charging_power = EV.max_ac_charge_power
            time_of_stay = (EV.time_of_departure -
                            EV.time_of_arrival)*timescale / 60

            EVs["arrival_time"].append(arrival_time)
            EVs["departure_time"].append(departure_time)
            EVs["SoC_at_arrival"].append(SoC_at_arrival)
            EVs["battery_capacity"].append(battery_capacity)
            EVs["charging_power"].append(charging_power)
            EVs["time_of_stay"].append(time_of_stay)

            # print(EVs)
            # exit()

        data = pd.DataFrame(EVs)
        all_data = pd.concat([all_data, data])

    # Create the figure for subplots
    # figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the distribution of SoC at arrival

    # plt.subplot(1, 3, 1)
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['font.family'] = ['serif']

    g0 = sns.JointGrid(data=all_data,
                       x="time_of_stay",
                       y="arrival_time",
                       hue="case",
                       xlim=(0, 20),
                       ylim=(200, 1600),
                       ratio=3,
                       height=4,                       
                       )

    # Plotting the joint KDE plot with transparency
    g0.plot_joint(sns.kdeplot, 
                  levels=20,
                #   gridsize=1000,
                  thresh=0.35,
                #   cut=5,
                fill=True,
                  alpha=0.6,
                #   cummulative=True,
                  common_grid=True,
                  common_norm=True,                  
                  )

    # Plotting the marginal KDE plots with transparency
    g0.plot_marginals(sns.kdeplot, alpha=1)
    
    g0.ax_joint.legend().remove()

    # plt.subplot(1, 3, 2)

    g1 = sns.JointGrid(data=all_data,
                       x="SoC_at_arrival",
                       y="arrival_time",
                       hue="case",
                       xlim=(0, 105),
                       ylim=(200, 1600),
                       ratio=3,
                       #  kind="kde",
                       #  palette="viridis",
                       height=4,
                       #  ratio=5
                       )

    # Plotting the joint KDE plot with transparency
    g1.plot_joint(sns.kdeplot, 
                  levels=20,
                #   gridsize=1000,
                  thresh=0.2,
                #   cut=5,
                fill=True,
                  alpha=0.6,
                #   cummulative=True,
                  common_grid=True,
                  common_norm=True,                  
                  )

    # Plotting the marginal KDE plots with transparency
    g1.plot_marginals(sns.kdeplot, alpha=1)

    # g1.ax_joint.legend()
    # remove the legend
    # g1.ax_joint.legend().remove()

    # add legend outside the plot

    g2 = sns.JointGrid(data=all_data,
                       x="battery_capacity",
                       y="arrival_time",
                       hue="case",
                       xlim=(20, 90),
                       ylim=(200, 1600),
                       height=4,
                       ratio=3,
                       #  kind="reg",
                       #  palette="Paired",
                       # height=6,
                       # ratio=5
                       )

    # Plotting the joint KDE plot with transparency
    g2.plot_joint(sns.kdeplot, 
                  levels=20,
                #   gridsize=1000,
                  thresh=0.3,
                #   cut=5,
                fill=True,
                  alpha=0.5,
                #   cummulative=True,
                  common_grid=True,
                  common_norm=True,                  
                  )

    # Plotting the marginal KDE plots with transparency
    g2.plot_marginals(sns.kdeplot, alpha=1)
    
    g2.ax_joint.legend().remove()
    g1.ax_joint.legend().remove()

    # sns.move_legend(g1.ax_joint,
    #                 "lower left",
    #                 title=None,
    #                 #number of columns
    #                 ncol=1,
    #                 frameon=True,
    #                 #width of the legend line
    #                 handlelength=1,
    #                 fontsize=11,)

    # remove the y-axis labels
    g0.ax_joint.set_ylabel('Arrival Time (hour)', fontsize=13)

    g0.ax_joint.set_yticks([0, 360, 720, 1080, 1440],
                           ['00:00', '06:00', '12:00', '18:00', '24:00'],
                           fontsize=12)

    g1.ax_joint.set_yticks([0, 360, 720, 1080, 1440],
                           ['', '', '', '', ''])

    g2.ax_joint.set_yticks([0, 360, 720, 1080, 1440],
                           ['', '', '', '', '']
                           )

    # add grid lines
    g0.ax_joint.grid(axis='y', linestyle='--', alpha=0.5)
    g1.ax_joint.grid(axis='y', linestyle='--', alpha=0.5)
    g2.ax_joint.grid(axis='y', linestyle='--', alpha=0.5)

    g0.ax_joint.grid(axis='x', linestyle='--', alpha=0.5)
    g1.ax_joint.grid(axis='x', linestyle='--', alpha=0.5)
    g2.ax_joint.grid(axis='x', linestyle='--', alpha=0.5)

#   change font size of x-ticks
    g0.ax_joint.set_xticks([0, 5, 10, 15, 20],
                           ['0', '5', '10', '15', '20'],
                           fontsize=12)
    g1.ax_joint.set_xticks([0, 20, 40, 60, 80, 100],
                           ['0', '20', '40', '60', '80', '100'],
                           fontsize=12)
    g2.ax_joint.set_xticks([20, 40, 60, 80],
                           ['20', '40', '60', '80'],
                           fontsize=12)

    g0.ax_joint.set_xlabel('Time of Stay (Hours)', fontsize=13)
    g1.ax_joint.set_ylabel('')
    g1.ax_joint.set_xlabel('SoC at Arrival (%)', fontsize=13)
    g2.ax_joint.set_ylabel('')
    g2.ax_joint.set_xlabel('Battery Capacity (kWh)', fontsize=13)

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 3)

    mg0 = SeabornFig2Grid(g0, fig, gs[0])
    mg1 = SeabornFig2Grid(g1, fig, gs[1])
    mg2 = SeabornFig2Grid(g2, fig, gs[2])

    g0.ax_marg_y.remove()
    g1.ax_marg_y.remove()

    # move the legend in ax_joint

    # set g1 legend outside the plot
    # g1.ax_joint.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # add custom legend under the image wit hthe four cases
    custom_lines = [matplotlib.lines.Line2D([0], [0], color='#1f77b4', lw=4),
                    matplotlib.lines.Line2D([0], [0], color='#ff8315', lw=4),
                    matplotlib.lines.Line2D([0], [0], color='#41a941', lw=4),
                    matplotlib.lines.Line2D([0], [0], color='#d62c2d', lw=4)]

    fig.legend(custom_lines, ['Original', 'Small', 'Medium', 'Extreme'],
               loc='lower center',
               ncol=5,
               fontsize=12,
               title='Case Variation:',
               title_fontsize=12,
               frameon=True,
               bbox_to_anchor=(0.5, 0.01),
               )

    plt.tight_layout()
    # plt.savefig("./Results_Analysis/scenario_distributions.png")
    plt.savefig("./Results_Analysis/scenario_distributions_1.png")

    plt.show()


if __name__ == "__main__":
    plot_optimality_gap_in_generalization()

    # plot_scenario_distributions()
