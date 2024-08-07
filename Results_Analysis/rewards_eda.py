import pandas as pd
import matplotlib.pyplot as plt

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

# PST 25
evse = 25
opt_mean = -2.0399
opt_std = 0.5546
upper_limit = -12.0000
lower_limit = -2.0000
epoch_limit = 2000
zoom_in_upper = -2.6000
zoom_in_lower = -4.0000

# PST 100
evse = 100
opt_mean = -10.8179
opt_std = 2.1453
upper_limit = -200.0000
lower_limit = -10
epoch_limit = 2000
zoom_in_upper = -12
zoom_in_lower = -50

# # PST 500
# evse = 500
# # opt_mean = -54.0895
# # opt_std = 10.7265
# upper_limit = -1900.0000
# lower_limit = -350
# epoch_limit = 1000
# zoom_in_upper = -60
# zoom_in_lower = -200

# ## PST 1000
# evse = 1000
# # opt_mean = -108.1790
# # opt_std = 21.4530
# upper_limit = -10000.0000
# lower_limit = -3500
# epoch_limit = 1000


# read multiple dfs and concatenate them
df_list = [
    # "./Results_Analysis/raw_data/SAC_25.csv",
    # "./Results_Analysis/raw_data/TD3_25.csv",
    # "./Results_Analysis/raw_data/SAC_100.csv",
    # "./Results_Analysis/raw_data/TD3_100.csv",
    # "./Results_Analysis/raw_data/SAC_500.csv",
    # "./Results_Analysis/raw_data/TD3_500.csv",
    # "./Results_Analysis/raw_data/SAC_1000.csv",
    # "./Results_Analysis/raw_data/TD3_1000.csv",
    "./Results_Analysis/raw_data/V2G_TD3_25.csv",
    "./Results_Analysis/raw_data/V2G_SAC_25.csv",
    # "./Results_Analysis/raw_data/V2G_TD3_500.csv",
    # "./Results_Analysis/raw_data/V2G_SAC_500.csv",

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

# drop Step row
df = df.drop('Step')

# divide columns by 10^4
df = df / 10_000

# create a new dataframe with grouped rows based on the first 3 characters of the index
df_mean = df.groupby(df.index.str[:5]).mean()


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
    '#16A085'   # Green Sea
]

plt.figure()
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = ['serif']

ax = plt.subplot(1, 2, 1)
for i in range(len(df_max)):
    plt.plot(df_max.iloc[i],
             marker=markers[i],
             markersize=3,
             linewidth=1.5,
             linestyle=line_styles[i],
             label=df_max.index[i],
             color=colors[i],
             markevery=100
             )

plt.ylim(upper_limit, lower_limit)
plt.xlim(-50, epoch_limit)


# change the y ticks to scientific notation
# plt.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
plt.yticks(fontsize=12)

plt.title('Best Eval. Model')
plt.xlabel('Epochs')
plt.ylabel('Reward ($ x10^4$)', fontsize=14)

if evse >= 500:
    plt.xticks([0, 250, 500, 750, 1000], fontsize=12)
else:
    plt.xticks([0, 500, 1000, 1500, 2000], fontsize=12)

# add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)


if evse < 500:
    # Make the zoom-in plot:
    axins = zoomed_inset_axes(ax,
                              2.8,
                              loc="lower right")

    start_epoch_zoom = 1500
    for i in range(len(df_max)):
        axins.plot(df_max.iloc[i][start_epoch_zoom:],
                   marker=markers[i],
                   #    markersize=3,
                   linewidth=1.5,
                   linestyle=line_styles[i],
                   color=colors[i],
                   markevery=100
                   )

    axins.set_xlim(start_epoch_zoom, epoch_limit)
    axins.set_ylim(zoom_in_lower, zoom_in_upper)

    plt.xticks(visible=False)
    plt.yticks(fontsize=10)
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


# plt.

# # plot the optimal line
# plt.axhline(opt_mean, xmin=0, xmax=epoch_limit,
#             color='b', linestyle='-', label='Optimal')
# plt.fill_between(df_mean.columns, opt_mean - opt_std,
#                  opt_mean + opt_std, color='r', alpha=0.2)


# plt.show()

# smooth the data using time weighteda Exponential Moving Average
# df_mean = df_mean.ewm(
#     alpha=0.5,
#     # span=10,
#     # com=10,
#     adjust=True).mean()

# show variation of rewards over time
df_var = df.groupby(df.index.str[:5]).std()

# show the maximum value of each group and the minimum value of each group
df_max = df.groupby(df.index.str[:5]).max()
df_min = df.groupby(df.index.str[:5]).min()


plt.subplot(1, 2, 2)
# plot every row
for i in range(len(df_mean)):
    plt.plot(df_mean.iloc[i],
             label=df_mean.index[i],
             #  markersize=3,
             linewidth=0.5,
             markevery=50,
             marker=markers[i],
             color=colors[i])

    plt.fill_between(df_mean.columns,
                     df_min.iloc[i],
                     df_max.iloc[i],
                     alpha=0.2)

plt.title('Average')

# plot the optimal line
# plt.axhline(opt_mean, xmin=-5000, xmax=epoch_limit,
#             color='k', linestyle='-', label='Optimal')
# plt.fill_between(df_mean.columns, opt_mean - opt_std,
#                  opt_mean + opt_std, color='gray', alpha=0.2)

plt.ylim(upper_limit, lower_limit)
plt.xlim(-50, epoch_limit)

plt.xlabel('Epochs')
# remove the y_tick labels but keep the grid
y_ticks = plt.gca().yaxis.get_major_ticks()
for t in y_ticks:
    t.label1.set_visible(False)
if evse >= 500:
    plt.xticks([0, 250, 500, 750, 1000], fontsize=12)
else:
    plt.xticks([0, 500, 1000, 1500, 2000], fontsize=12)

# add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# plt.xlim(100)
# plt.legend()
plt.show()
