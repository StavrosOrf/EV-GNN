import pandas as pd


data = pd.read_csv(
    './results/eval_1000cs_70tr_PublicPST_1000_13_algos_100_exp_2024_07_30_508748/data.csv')
data2 = pd.read_csv(
    './results/eval_1000cs_70tr_PublicPST_1000_1_algos_100_exp_2024_07_30_392080/data.csv')

# concatenate the dataframes
data = pd.concat([data, data2], axis=0)

# group by algotithm and get mean and std
columns = ['Unnamed: 0', 'run', 'Algorithm', 'total_ev_served', 'total_profits',
           'total_energy_charged', 'total_energy_discharged',
           'average_user_satisfaction', 'power_tracker_violation',
           'tracking_error', 'energy_tracking_error', 'energy_user_satisfaction',
           'total_transformer_overload', 'battery_degradation',
           'battery_degradation_calendar', 'battery_degradation_cycling',
           'total_reward']

columns_to_keep = ['Algorithm', 'run',
                   'total_energy_charged',
                   'average_user_satisfaction',
                   # 'tracking_error',
                   'energy_tracking_error',
                   'time',
                   'total_reward']
data = data[columns_to_keep]

# find the PowerTrackingErrorrMin.total_reward
PowerTrackingErrorrMin = data[data.Algorithm ==
                              "PowerTrackingErrorrMin"]
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

columns_to_drop = [
    'run',
    'total_reward',
    'G_E',
    'time'
]

data = data.drop(columns=columns_to_drop)


data_grouped = data.groupby('Algorithm').agg(['mean', 'std'])

# create new columns with the mean and std of the total_energy_charged combined as a string
data_grouped['total_energy_charged'] = data_grouped['total_energy_charged']\
    .apply(lambda x: f"${x['mean']/1000:.1f}$ ±${x['std']/1000:.1f}$", axis=1)
data_grouped['average_user_satisfaction'] = data_grouped['average_user_satisfaction']\
    .apply(lambda x: f"${x['mean']*100:.1f}$ ±${x['std']*100:.1f}$", axis=1)
# data_grouped['tracking_error'] = data_grouped['tracking_error']\
#        .apply(lambda x: f"${x['mean']/1000:.1f}$ ±${(x['std']/1000):.1f}$", axis=1)
data_grouped['energy_tracking_error'] = data_grouped['energy_tracking_error']\
    .apply(lambda x: f"${x['mean']/1000:.0f}$ ±${x['std']/1000:.1f}$", axis=1)
# data_grouped['total_reward'] = data_grouped['total_reward']\
#        .apply(lambda x: f"${x['mean']/1000:.1f}$ ±${x['std']/1000:.1f}$", axis=1)
data_grouped['G'] = data_grouped['G']\
    .apply(lambda x: f"${x['mean']:.0f}$ ±${x['std']:.0f}$", axis=1)
# data_grouped['time'] = data_grouped['time']\
#        .apply(lambda x: f"${x['mean']/112:.1f}$ ±${x['std']/112:.1f}$", axis=1)

# rearange rows


# drop the mean and std columns
data_grouped = data_grouped.droplevel(1, axis=1)
# print the results
# drop duplicate columns
data_grouped = data_grouped.loc[:, ~data_grouped.columns.duplicated()]
# rename columns
data_grouped.columns = ['Energy Charged (MWh)',
                        'User Satisfaction ()',
                        'Energy Error (MWh)',
                        #    'Step time (s)',
                        'G ()']


# rename algorithm names with shorter names
data_grouped.index = data_grouped.index.str.replace(
    'PowerTrackingErrorrMin', 'Optimal')
data_grouped.index = data_grouped.index.str.replace(
    'ChargeAsFastAsPossible', 'AFAP')
data_grouped.index = data_grouped.index.str.replace('RoundRobin', 'RR')
data_grouped.index = data_grouped.index.str.replace(
    'SAC_ActionGNN', 'SAC EV-GNN')
data_grouped.index = data_grouped.index.str.replace('SAC_GNN', 'SAC GNN-FX')

data_grouped.index = data_grouped.index.str.replace(
    'TD3_ActionGNN', 'TD3 EV-GNN')
data_grouped.index = data_grouped.index.str.replace('TD3_GNN', 'TD3 GNN-FX')
data_grouped.index = data_grouped.index.str.replace('ppo', 'PPO')
data_grouped.index = data_grouped.index.str.replace('sac', 'SAC')
data_grouped.index = data_grouped.index.str.replace('td3', 'TD3')
data_grouped.index = data_grouped.index.str.replace('ddpg', 'DDPG')
data_grouped.index = data_grouped.index.str.replace('a2c', 'A2C')
data_grouped.index = data_grouped.index.str.replace('tqc', 'TQC')
data_grouped.index = data_grouped.index.str.replace('trpo', 'TRPO')


# change order of rows
data_grouped = data_grouped.reindex(['AFAP',
                                     'RR',
                                     'A2C',
                                     'DDPG',
                                     'PPO',
                                     'TRPO',
                                     'TQC',
                                     'SAC',
                                     'TD3',
                                     'SAC GNN-FX',
                                     'TD3 GNN-FX',
                                     'SAC EV-GNN',
                                     'TD3 EV-GNN',

                                     ])


# rename PowerTrackingErrorrMin to Optimal
# print(data_grouped)
print(data_grouped.to_latex())
