import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import time

from matplotlib import pyplot as plt


# 25 CS, 3 TR, V2G, ProfixMax, WithLoads
# data = pd.read_csv('./results/eval_25cs_3tr_V2G_ProfixMaxWithLoads_25_11_algos_3_exp_2024_07_18_418263/data.csv')
# data1 = pd.read_csv('./results/eval_500cs_35tr_V2G_ProfixMaxWithLoads_500_11_algos_5_exp_2024_07_21_407107/data.csv')
data1 = pd.read_csv('./results/eval_500cs_35tr_V2G_ProfixMaxWithLoads_500_9_algos_100_exp_2024_07_23_584957/data.csv')
# data = pd.read_csv('./results/eval_25cs_3tr_V2G_ProfixMaxWithLoads_25_11_algos_3_exp_2024_07_18_418263/data.csv')


# drop rows were Algorithm is eMPC_v2G
data1 = data1[data1['Algorithm'] != 'eMPC_V2G_v2']


data2 = pd.read_csv('./results/eval_500cs_35tr_V2G_ProfixMaxWithLoads_500_9_algos_5_exp_2024_07_23_701565/data.csv')
data = pd.concat([data1, data2], axis=0)
print(data.columns)


columns_to_keep = [
    # 'run',
    'Algorithm',
    'total_profits',
    'total_energy_charged',
    'total_energy_discharged',
    'average_user_satisfaction',
    'energy_user_satisfaction',
    'total_transformer_overload',
    'time',
]

data = data[columns_to_keep]


# multiply by 100 average_user_satisfaction
data['average_user_satisfaction'] = data['average_user_satisfaction'] * 100
data['total_energy_charged'] = data['total_energy_charged'] / 1000
data['total_energy_discharged'] = -data['total_energy_discharged'] / 1000
data['total_transformer_overload'] = data['total_transformer_overload'] / 1000


data['Algorithm'] = data['Algorithm'].replace({"<class 'ev2gym.baselines.heuristics.ChargeAsFastAsPossible'>": 'AFAP',
                                               "ChargeAsFastAsPossible": 'AFAP',
                                               "RoundRobin": 'RR',
                                               "<class 'ev2gym.baselines.heuristics.RoundRobin_GF'>": 'RR_GF',
                                               "<class 'ev2gym.baselines.heuristics.RoundRobin'>": 'RR',
                                               "<class 'ev2gym.baselines.gurobi_models.tracking_error.PowerTrackingErrorrMin'>": 'Optimal',
                                               "PowerTrackingErrorrMin": 'Optimal',
                                               "V2GProfitMaxOracleGB": 'Optimal',
                                               "eMPC_V2G_v2": 'MPC',
                                               }
                                              )

# change Algorithm if string is in Algorithm
data['Algorithm'] = data['Algorithm'].apply(
    lambda x: 'SAC GNN-FX' if 'SAC_GNN' in x else x)
data['Algorithm'] = data['Algorithm'].apply(
    lambda x: 'SAC EV-GNN' if 'SAC_ActionGNN' in x else x)
data['Algorithm'] = data['Algorithm'].apply(
    lambda x: 'SAC' if 'SAC_run' in x else x)

data['Algorithm'] = data['Algorithm'].apply(
    lambda x: 'TD3 GNN-FX' if 'TD3_GNN' in x else x)
data['Algorithm'] = data['Algorithm'].apply(
    lambda x: 'TD3 EV-GNN' if 'TD3_ActionGNN' in x else x)
data['Algorithm'] = data['Algorithm'].apply(
    lambda x: 'TD3' if 'TD3_run' in x else x)

data = data[data['Algorithm'] != 'ChargeAsLateAsPossible']

plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = ['serif']

plt.figure(figsize=(10, 4))

print(data.Algorithm.unique())

#divide the values of column time by 1000 to get the time in seconds
data['time'] = data['time'] / 96

# 5 subplots
plt.subplot(1, 5, 1)

sns.boxplot(x='Algorithm',
            y='total_profits',
            showfliers=False,
            order=['AFAP', 'RR',
                   'SAC', 'TD3',
                   'SAC GNN-FX', 'TD3 GNN-FX',
                   'SAC EV-GNN', 'TD3 EV-GNN',
                   'MPC', 'Optimal'],
            hue='Algorithm',
            data=data)

# remove x-axis label and x-ticks
plt.xlabel('')

# add a horizontal line at y=0
plt.axhline(y=0, color='black', linewidth=1, linestyle='--')

# keep x-tick ticks but witthout the labels
pos = np.arange(len(data['Algorithm'].unique()))
plt.xticks(pos, ['' for i in range(len(data['Algorithm'].unique()))])

# add grid
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.grid(axis='x', linestyle='--', alpha=0.2)

plt.ylabel('Total Profits (€)', fontsize=12)
plt.yticks(fontsize=11)

plt.subplot(1, 5, 2)

sns.boxplot(x='Algorithm',
            y='average_user_satisfaction',
            hue='Algorithm',
            showfliers=False,
            order=['AFAP', 'RR',
                   'SAC', 'TD3',
                   'SAC GNN-FX', 'TD3 GNN-FX',
                   'SAC EV-GNN', 'TD3 EV-GNN',
                   'MPC', 'Optimal'],
            data=data)
# remove x-axis label and x-ticks
plt.xlabel('')
plt.ylabel('user Satisfaction (%)', fontsize=12)
plt.yticks(fontsize=11)

# keep x-tick ticks but witthout the labels
pos = np.arange(len(data['Algorithm'].unique()))
plt.xticks(pos, ['' for i in range(len(data['Algorithm'].unique()))])

# add grid
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.grid(axis='x', linestyle='--', alpha=0.2)

ax1 = plt.subplot(1, 5, 3)

ax = sns.boxplot(x='Algorithm',
                 y='total_transformer_overload',
                 hue='Algorithm',
                 legend='full',
                 showfliers=False,
                 order=['AFAP', 'RR',
                        'SAC', 'TD3',
                        'SAC GNN-FX', 'TD3 GNN-FX',
                        'SAC EV-GNN', 'TD3 EV-GNN',
                        'MPC', 'Optimal'],
                 data=data)
# remove x-axis label and x-ticks
plt.xlabel('')
plt.ylabel('Trasnformer Overloads (MWh)', fontsize=12)
plt.yticks(fontsize=11)

# keep x-tick ticks but witthout the labels
pos = np.arange(len(data['Algorithm'].unique()))
plt.xticks(pos, ['' for i in range(len(data['Algorithm'].unique()))])

# add grid
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.grid(axis='x', linestyle='--', alpha=0.2)

handles, labels = ax.get_legend_handles_labels()
print(labels)
print(handles)
# labels = ['AFAP' 'SAC GNN-FX' 
# 'SAC' 'SAC EV-GNN'
# 'TD3' 'TD3 GNN-FX'
# 'TD3 EV-GNN'
#  'RR'
# 'Optimal' 'MPC']

# reorder the labels and handles to match the order of the boxplot
labels = ['AFAP', 'RR',
          'SAC', 'TD3',
          'SAC GNN-FX', 'TD3 GNN-FX',
          'SAC EV-GNN', 'TD3 EV-GNN',
          'MPC', 'Optimal']

handles = [handles[0], handles[7],
           handles[2], handles[4],
           handles[1], handles[5],
           handles[3], handles[6],
           handles[9], handles[8]]


# # Then, create a legend outside the plot area
ax1.legend(handles=handles, labels=labels,
           title='',
           loc='upper center',
           bbox_to_anchor=(0.5, -0.05),
           ncol=5)


plt.subplot(1, 5, 4)


sns.barplot(x='Algorithm',
            y='total_energy_charged',
            data=data,
            hue='Algorithm',
            order=['AFAP', 'RR',
                   'SAC', 'TD3',
                   'SAC GNN-FX', 'TD3 GNN-FX',
                   'SAC EV-GNN', 'TD3 EV-GNN',
                   'MPC', 'Optimal'],
            ci='sd',
            )

sns.barplot(x='Algorithm',
            y='total_energy_discharged',
            data=data,
            hue='Algorithm',
            order=['AFAP', 'RR',
                   'SAC', 'TD3',
                   'SAC GNN-FX', 'TD3 GNN-FX',
                   'SAC EV-GNN', 'TD3 EV-GNN',
                   'MPC', 'Optimal'],
            ci='sd')

# add a horizontal line at y=0
plt.axhline(y=0, color='black', linewidth=1, linestyle='--')

# remove x-axis label and x-ticks
plt.xlabel('')
plt.ylabel('Energy (Dis)-Charged (MWh)', fontsize=12)
plt.yticks(fontsize=11)

# keep x-tick ticks but witthout the labels
pos = np.arange(len(data['Algorithm'].unique()))
plt.xticks(pos, ['' for i in range(len(data['Algorithm'].unique()))])

# add grid
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.grid(axis='x', linestyle='--', alpha=0.2)

plt.subplot(1, 5, 5)

sns.boxplot(x='Algorithm',
            y='time',
            hue='Algorithm',
            order=['AFAP', 'RR',
                   'SAC', 'TD3',
                   'SAC GNN-FX', 'TD3 GNN-FX',
                   'SAC EV-GNN', 'TD3 EV-GNN',
                   'MPC', 'Optimal'],
            showfliers=False,
            # orient='h',
            data=data)

# remove x-axis label and x-ticks
plt.xlabel('')
plt.ylabel('Step Execution Time (s)', fontsize=12)
plt.yticks(fontsize=11)

# make log scale
plt.yscale('log')

# keep x-tick ticks but witthout the labels
pos = np.arange(len(data['Algorithm'].unique()))
plt.xticks(pos, ['' for i in range(len(data['Algorithm'].unique()))])

# add grid
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.grid(axis='x', linestyle='--', alpha=0.2)

# plt.tight_layout()
plt.savefig('./Results_Analysis/v2g_stats_eda.png')

plt.show()
