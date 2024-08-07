import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker
import numpy as np

EV_list = [
    25,
    100,
    # 500,
    # 1000,

]
# avg of 100 runs
optimal_mean = [
    -20_556,
    -108_539,
    -2_764_761,
    -11_034_490
]

optimal_std = [
    6_191,
    24_080,
    248_145,
    899_527
]

all_data = pd.DataFrame(
    columns=["Algorithm", "EV_num", "Optimality_Gap_mean", "Optimality_Gap_low", "Optimality_Gap_high"])

for i, EV_num in enumerate(EV_list):
    # Load the data
    data = pd.read_csv(f'./Results_Analysis/raw_data/SB3_Discr_{EV_num}.csv')
    data_TD3 = pd.read_csv(
        f'./Results_Analysis/raw_data/TD3_Discr_{EV_num}.csv')

    # concatenate the dataframes
    data = pd.concat([data, data_TD3])

    # drop columns that have the word MIN and MAX
    data = data.loc[:, ~data.columns.str.contains('MIN')]
    data = data.loc[:, ~data.columns.str.contains('MAX')]
    # print(f'Columns: {data.columns}')

    # find the maximum value for each column
    max_values = data.max()
    # drop column 'Step' from the max_values
    max_values = max_values.drop('Step')

    # change name of the index to 'algorithm' by keeping the first two split() of the column name
    max_values.index = max_values.index.str.split('_').str[:2].str.join('_')

    # if the column name has "_SimpleReward" remove it
    max_values.index = max_values.index.str.replace('_SimpleReward', '')
    max_values.index = max_values.index.str.replace('_run', '')

    # print(f'Maximum values: {max_values}')

    # keep the two larger values grouping by the first split of the column name

    max_values = max_values.groupby(max_values.index).nlargest(1)
    # print(f'Two largest values: {max_values}')

    # make new dataframe with columns "Algorithm","EV_num", "Optimality_Gap_mean", "Optimality_Gap_std

    # iterate over the max_values
    for idx, value in max_values.items():
        # split the index
        algorithm = idx[0]
        EV = EV_num
        # calculate the mean and std of the optimality gap
        mean = abs(optimal_mean[i] - value) / abs(optimal_mean[i]) * 100

        std_low = abs(optimal_mean[i] - value -
                      optimal_std[i]) / abs(optimal_mean[i]) * 100

        std_high = abs(optimal_mean[i] - value +
                       optimal_std[i]) / abs(optimal_mean[i]) * 100

        std_low = abs(mean - std_low)
        std_high = abs(mean - std_high)

        # append the values to the dataframe

        # use concat
        all_data = pd.concat([all_data, pd.DataFrame([[algorithm, EV, mean, std_low, std_high]], columns=[
                             "Algorithm", "EV_num", "Optimality_Gap_mean",
                             "Optimality_Gap_low", "Optimality_Gap_high"
                             ])])

#reset the index
all_data.reset_index(drop=True, inplace=True)
print(all_data)

# change the Algorithm names
all_data['Algorithm'] = all_data['Algorithm'].replace(
    {'a2c': 'A2C',
     'ddpg': 'DDPG',
        'ppo': 'PPO\n(Normal)',
        'trpo': 'TRPO',
        'tqc': 'TQC',
        'SAC_GNN': 'SAC\nGNN-FX',
        'TD3_GNN': 'TD3\nGNN-FX',
        'SAC_ActionGNN': 'SAC\nEV-GNN',
        'TD3_ActionGNN': 'TD3\nEV-GNN\n(Multi-Discrete)',
        'MaskablePPO': 'Mask\nPPO',
        'RecurrentPPO': 'Recurrent\nPPO'
     })

# plot the data using seaborn


sns.set_theme(style="whitegrid")

# Draw a nested barplot by species and

# figure size
fig, ax = plt.subplots(figsize=(5, 4))
plt.rcParams['font.family'] = ['serif']
print(all_data)
ax_g = sns.catplot(ax=ax,
                   data=all_data,
                   kind="bar",
                   order=['A2C', 'TRPO', 'PPO\n(Normal)', 'Mask\nPPO',
                          'Recurrent\nPPO',
                          'TD3\nEV-GNN\n(Multi-Discrete)'],
                   x="Algorithm",
                   y="Optimality_Gap_mean",
                   hue="EV_num",
                   ci="sd",
                   palette="dark",
                   alpha=0.9,
                   saturation=1,
                   height=5,
                   aspect=2,
                   legend_out=False,
                   )

# errors = (all_data.Optimality_Gap_low, all_data.Optimality_Gap_high)
errors = {}

algorithms = ['A2C', 'TRPO', 'PPO', 'Mask\nPPO',
              'Recurrent\nPPO',
              'TD3\nEV-GNN\n(Multi-Discrete)']

# for algo in algorithms:
#     for ev_number in [25, 100]:

#         key = (algo, ev_number)
#         lower_error = all_data.loc[(all_data['Algorithm'] == algo) & (
#             all_data['EV_num'] == ev_number), 'Optimality_Gap_low'].values[0]
#         upper_error = all_data.loc[(all_data['Algorithm'] == algo) & (
#             all_data['EV_num'] == ev_number), 'Optimality_Gap_high'].values[0]

#         errors[key] = (lower_error, upper_error)

#         # print(f'Errors: {errors}')

ax_c = ax_g.ax


# # Overlay custom error bars
# for i, bar in enumerate(ax_c.patches):
#     category = algorithms[i % 11]
#     ev_num = all_data['EV_num'].unique()[i % 4]

#     if (category, ev_num) in errors:
#         bar_center = bar.get_x() + bar.get_width() / 2
#         bar_height = bar.get_height()

#         lower_error, upper_error = errors[(category, ev_num)]

#         print(f'Category: {category}, EV: {ev_num}, Lower: {lower_error}, Upper: {upper_error}')

#         ax_c.errorbar(bar_center, bar_height, yerr=[[lower_error], [upper_error]],
#                       fmt='o', color='black', capsize=3)

# connect the error bars for each algorithm with a line


# add vertical line at x=5.5
plt.axvline(x=4.5, color='black', linestyle='--', linewidth=1)


# show the tick small dashes
plt.tick_params(axis='x', which='both', bottom=True, top=False)
plt.tick_params(axis='y', which='both', left=True, right=False)

plt.legend(title='Number of EVSE', loc='upper center', ncol=2,
           shadow=True,
           fontsize=13)
plt.ylabel('Mean Optimality Gap (%)', fontsize=14)
plt.xlabel('')
plt.xticks(
    # rotation=45,
    fontsize=14)
# use logaritmic scale
# plt.yscale('log',
#            base=10,
#            )


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

# plt.ylim(10, 5_000)
plt.ylim(10, 2_000)
# plt.ylim(10, 5_000)
plt.yticks(fontsize=13)


plt.tight_layout()
plt.savefig('./Results_Analysis/comp_opt_gap_pst.png')
plt.show()
