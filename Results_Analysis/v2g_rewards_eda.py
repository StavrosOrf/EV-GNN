import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


# load wandb_export_2024-06-18T13_30_46.429+02_00.csv as df

# df = pd.read_csv('./Results_Analysis/raw_data/SAC_1000.csv')
# df = pd.read_csv('./Results_Analysis/raw_data/TD3_1000.csv')
# df = pd.read_csv('./Results_Analysis/raw_data/TD3_500.csv')
# df = pd.read_csv('./Results_Analysis/raw_data/SAC_100.csv')
# df = pd.read_csv('./Results_Analysis/raw_data/TD3_100.csv')
# df = pd.read_csv('./Results_Analysis/raw_data/TD3_25.csv')
# df = pd.read_csv('./Results_Analysis/raw_data/SAC_25.csv')

# V2G 500
evse = 25

upper_limit = 100
lower_limit = -50
epoch_limit = 600
zoom_in_upper = 0.1
zoom_in_lower = -1.5

# V2G 500
evse = 500

upper_limit = 0.1
lower_limit = -2
epoch_limit = 600
zoom_in_upper = 0.1
zoom_in_lower = -1.5


# read multiple dfs and concatenate them
df_list = [
    # "./Results_Analysis/raw_data/V2G_SAC_25.csv",
    # "./Results_Analysis/raw_data/V2G_TD3_25.csv",

    "./Results_Analysis/raw_data/V2G_TD3_500.csv",
    "./Results_Analysis/raw_data/V2G_SAC_500.csv",
]


df = pd.read_csv(df_list[0])
for df_path in df_list[1:]:
    df = pd.concat([df, pd.read_csv(df_path)], axis=1)

# print(df.columns)

# drop columns containing min and max in the name

for col in df.columns:
    if 'MIN' in col or 'MAX' in col:
        df.drop(col, axis=1, inplace=True)
# print(df.columns)

# transpose the dataframe
df = df.T

# keep only the epoch_limit rows
df = df.iloc[:, :epoch_limit]

# drop Step row
df = df.drop('Step')

# divide columns by 10^4
if evse > 25:
    df = df / 10_000

# create a new dataframe with grouped rows based on the first 3 characters of the index
df_mean = df.groupby(df.index.str[:5]).mean()

# print(f'df_mean: {df_mean}')

# create df_max which contains the maximum value so far of each group

df_max = df.groupby(df.index.str[:5]).cummax(axis=1)

# print the maximum value of each group
max_names = df_max.max(axis=1)


df_max = df_max.groupby(df_max.index.str[:5]).max().cummax(axis=1)

# keep the first 2000 episodes

df_mean = df_mean.iloc[:, :epoch_limit]
df_max = df_max.iloc[:, :epoch_limit]
# df_min = df_min.iloc[:, :2000]
df = df.iloc[:, :epoch_limit]

for i in range(len(df_max)):
    run_name = max_names[max_names == df_max.iloc[i].max()].index
    print(f'{df_max.index[i]}: {df_max.iloc[i].max()} {run_name}')


# extend df_max so that every row has the same length (epoch_limit)
# df_max = df_max.apply(lambda x: pd.concat([x,
#                                            pd.Series([x.iloc[-1]]*(epoch_limit - len(x)))],
#                                           ignore_index=True))


# optimal line

markers = ['o', 's', 'D', '^', '*', 'p', 'h', 'H', 'd', 'P', 'X']
line_styles = ['-', '--', '-.', ':', '-', '--', '-.',
               ':', '-', '--', '-.', ':', '-', '--', '-.', ':']

# colors =["#6C8CE6",
#          "#E6C36C",
#          "#6CE677",
#          "#9B9994",
#          "#66555B",
#          "#E65A89"]


colors = [
    '#45B7D1',  # Sky Blue
    '#E74C3C',  # Bright Red
    '#2C3E50',  # Dark Blue (Slate)
    '#F39C12',  # Amber
    '#8E44AD',  # Wisteria Purple
    '#16A085',   # Green Sea
    # '#16A085',   # Green Sea
    # '#16A085',   # Green Sea
]

plt.figure(figsize=(5,3))

plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = ['serif']
plt.title(f'{evse} EVSEs')

#place the title a bit higher 
plt.title(f'{evse} EVSEs', y=1.05)

# remove border and ticks
plt.tick_params(axis='both', which='both', bottom=False, top=False,
                labelbottom=False, left=False, right=False, labelleft=False)

# remove the right and top spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)


ax = plt.subplot(1, 2, 1)

print("df_max columns")
print(df_max.T.shape)
df_max = df_max.T

print(f'df_max: {df_max}')


# feel NaN values with the previous value
df_max.fillna(method='ffill', inplace=True, axis=0)

print("\n \n df_max columns \n \n")
print(f'{df_max}')

# replace the column names with     replace_legend = {
    #     'SAC_r': 'SAC',
    #     'SAC_G': 'SAC GNN-FX',
    #     'SAC_A': 'SAC EV-GNN',
    #     'TD3_r': 'TD3',
    #     'TD3_G': 'TD3 GNN-FX',
    #     'TD3_A': 'TD3 EV-GNN',
    # }

df_max.columns = df_max.columns.str.replace('SAC_r', 'SAC')
df_max.columns = df_max.columns.str.replace('SAC_G', 'SAC GNN-FX')
df_max.columns = df_max.columns.str.replace('SAC_A', 'SAC EV-GNN')
df_max.columns = df_max.columns.str.replace('TD3_r', 'TD3')
df_max.columns = df_max.columns.str.replace('TD3_G', 'TD3 GNN-FX')
df_max.columns = df_max.columns.str.replace('TD3_A', 'TD3 EV-GNN')


sns.lineplot(data=df_max,
             dashes=False,
             markers=markers[:6],
             markevery=100,
             markersize=5,
             alpha=0.6,
             legend='auto',
             palette=colors)


# remove legend
if evse > 25:

    plt.legend().remove()
    # plt.legend(loc='lower right')
else:
    plt.legend(loc='lower right', fontsize=9.5)
    

if evse > 25:
    plt.xlim(0, epoch_limit)    
    plt.ylim(-12,1)
else:
    plt.xlim(0, epoch_limit)    
    plt.ylim(-400,75)


# change the y ticks to scientific notation
# plt.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
plt.yticks(fontsize=12)

plt.title('Best')
plt.xlabel('Epochs')


if evse > 25:
    plt.ylabel('Reward ($ x10^4$)', fontsize=14)
else:
    plt.ylabel('Reward', fontsize=14)

# if evse > 25:
#     # plt.xticks(fontsize=12)
#     # make y-ticks scientific notation
#     plt.ticklabel_format(axis='y',
#                         style='sci',
#                         scilimits=(1, 4))

# if evse >= 500:
#     plt.xticks([0, 250, 500, 750, 1000], fontsize=12)
# else:
#     plt.xticks([0, 500, 1000, 1500, 2000], fontsize=12)

# add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)


if evse > 25:
    # Make the zoom-in plot:
    axins = zoomed_inset_axes(ax,
                            3.5,
                            loc="center right")

    start_epoch_zoom = 470


    sns.lineplot(data=df_max,
                dashes=False,
                markers=markers[:len(df_max)],
                markevery=100,
                alpha=0.6,
                palette=colors)

    plt.legend().remove()


    axins.set_xlim(start_epoch_zoom, epoch_limit)
    axins.set_ylim(zoom_in_lower, zoom_in_upper)



    plt.xticks(visible=False)
    plt.yticks(fontsize=11)
    mark_inset(ax, axins,
            loc1=1,
            loc2=2,
            fc="0.92",
            ec="0.5",
            lw=1)

    # add grid
    plt.grid(True, which='Major', linestyle='--', linewidth=0.5)

    # plt.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
    plt.yticks(fontsize=12)

    plt.draw()


plt.subplot(1, 2, 2)
# plt.savefig(f'./Results_Analysis/V2G_{evse}_rewards.png', dpi=300)
# exit()
# refactor the name of df columns to only have the first 5 characters
df = df.T
print(f'df columns: {df.columns}')
df.columns = df.columns.str[:5]
print(f'df columns: {df.columns}')
print(df)

#         SAC_A     SAC_A     SAC_A     SAC_A  ...     TD3_G     TD3_G     TD3_G     TD3_G0   -2.668070 -0.588094 -0.007367 -1.972475  ... -0.004424 -0.012397 -0.018379 -0.0052871   -0.011198 -0.011621 -0.011406 -0.011309  ... -0.057924 -0.082627 -0.145475 -0.0317022   -0.010989 -0.011141 -0.011262 -0.011357  ... -0.154283 -0.015360 -0.157026 -0.0132093   -0.010486 -0.011076 -0.011096 -0.011337  ... -0.179049 -0.011972 -0.370270 -0.0115074   -0.011357 -0.032649 -0.011232 -0.011195  ... -0.346751 -0.019789 -0.491193 -0.011264..        ...       ...       ...       ...  ...       ...       ...       ...       ...595 -0.001045 -0.004178 -0.003401  0.000201  ... -0.027991 -0.021085 -0.162808 -0.011498596 -0.003120 -0.005465 -0.008997 -0.002369  ... -0.040793 -0.178279 -0.243296 -0.011142597  0.001347 -0.004843 -0.007574 -0.002661  ... -0.012600 -0.015696 -0.159085 -0.031620598 -0.002927 -0.002579 -0.004216 -0.004697  ... -0.031505 -0.104768 -0.171164 -0.031441599 -0.002498 -0.000780 -0.002586 -0.006030  ... -0.011371 -0.057188 -0.364895 -0.011315

# plot the mean of the rewards for each group using df
df.index.name = 'Timestep'
df_long = df.reset_index().melt(id_vars='Timestep',
                                var_name='Algorithm',
                                value_name='Value')

print(f'df_long: {df_long}')

sns.lineplot(data=df_long,
             x='Timestep',
             y='Value',
             hue='Algorithm',
             dashes=False,
             markers=markers[:6],
             markevery=100,
             markersize=5,
             alpha=0.6,
             palette=colors)

plt.legend().remove()

# remove y-axis label 
plt.ylabel('')

if evse > 25:
    plt.xlim(0, epoch_limit)    
    plt.ylim(-50,2)
else:
    plt.xlim(0, epoch_limit)    
    plt.ylim(-3500, 100)

# if evse > 25:
#     # plt.xticks(fontsize=12)
#     # make y-ticks scientific notation
#     plt.ticklabel_format(axis='y',
#                         style='sci',
#                         scilimits=(4, 4))
    
plt.title('Average')


plt.xlabel('Epochs')

# add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig(f'./Results_Analysis/V2G_{evse}_rewards.png', dpi=300)
plt.show()
