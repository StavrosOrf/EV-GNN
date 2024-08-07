import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


EV_list = [
    25,
    100,
    500,
    1000,
]

plt.figure(figsize=(6, 7))
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = ['serif']
    
for i_index, EV_num in enumerate(EV_list):
    # Load the data    
    data_SAC = pd.read_csv(f'./Results_Analysis/raw_data/SAC_{EV_num}.csv')
    data_TD3 = pd.read_csv(f'./Results_Analysis/raw_data/TD3_{EV_num}.csv')

    # concatenate the dataframes
    data = pd.concat([data_SAC, data_TD3],axis=1)

    for col in data.columns:
        if 'MIN' in col or 'MAX' in col:
            data.drop(col, axis=1, inplace=True)
    
    data = data.T

    # # find the maximum value for each column
    # max_values = data.max()
    # # drop column 'Step' from the max_values
    data = data.drop('Step')

    # change name of the index to 'algorithm' by keeping the first two split() of the column name
    data.index = data.index.str.split('_').str[:2].str.join('_')
    
    #rename the Algorithm column
    data.index = data.index.str.replace('_SimpleReward', '')
    data.index = data.index.str.replace('_run', '')
    
        #    # change Algorithm if string is in Algorithm
        # data['Algorithm'] = data['Algorithm'].apply(
        #     lambda x: '  SAC  \nGNN-FX' if 'SAC_GNN' in x else x)
        # data['Algorithm'] = data['Algorithm'].apply(
        #     lambda x: '  SAC  \nEV-GNN' if 'SAC_ActionGNN' in x else x)
        # data['Algorithm'] = data['Algorithm'].apply(
        #     lambda x: 'SAC' if 'SAC_run' in x else x)

        # data['Algorithm'] = data['Algorithm'].apply(
        #     lambda x: '  TD3  \nGNN-FX' if 'TD3_GNN' in x else x)
        # data['Algorithm'] = data['Algorithm'].apply(
        #     lambda x: '  TD3  \nEV-GNN' if 'TD3_ActionGNN' in x else x)
        # data['Algorithm'] = data['Algorithm'].apply(
        #     lambda x: 'TD3' if 'TD3_run' in x else x)
        
    # rename the index to 'Algorithm'
    print(data)
    
    # if index contains 'SAC_ActionGNN' then rename to 'SAC'
    data.index = data.index.str.replace('SAC_ActionGNN', 'SAC EV-GNN')
    data.index = data.index.str.replace('SAC_GNN', 'SAC GNN-FX')
    data.index = data.index.str.replace('SAC_run', 'SAC')
    
    data.index = data.index.str.replace('TD3_ActionGNN', 'TD3 EV-GNN')
    data.index = data.index.str.replace('TD3_GNN', 'TD3 GNN-FX')
    data.index = data.index.str.replace('TD3_run', 'TD3')
    
    # add column with the EV number
    # data['EV_num'] = int(EV_num)

    df= data 
    
    # fill the NaN values with the previous value
    df = df.fillna(method='ffill', axis=1)

    # divide columns by 10^4
    # df = df / 10_000

    # create a new dataframe with grouped rows based on the first 3 characters of the index
    df_mean = df.groupby(df.index).mean()


    # create df_max which contains the maximum value so far of each group

    df_max = df.groupby(df.index).cummax(axis=1)

    # print the maximum value of each group
    max_names = df_max.max(axis=1)


    df_max = df_max.groupby(df_max.index).max().cummax(axis=1)

    # keep the first 2000 episodes

    # df_mean = df_mean.iloc[:, :epoch_limit]
    # df_max = df_max.iloc[:, :epoch_limit]    
    # df = df.iloc[:, :epoch_limit]

    for i in range(len(df_max)):
        run_name = max_names[max_names == df_max.iloc[i].max()].index
        print(f'{df_max.index[i]}: {df_max.iloc[i].max()} {run_name}')

    # optimal line

    markers = ['o', 's', 'D', '^', '*', 'p', 'h', 'H', 'd', 'P', 'X']
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.',
                ':', '-', '--', '-.', ':', '-', '--', '-.', ':']

    # reorder the rows
    df_max = df_max.reindex(['SAC', 'TD3',
                            'SAC GNN-FX', 'TD3 GNN-FX',
                            'SAC EV-GNN', 'TD3 EV-GNN'])


    colors = [
        '#16A085',   # Green Sea
        '#E74C3C',  # Bright Red
        '#2C3E50',  # Dark Blue (Slate)
        '#8E44AD',  # Wisteria Purple
        '#45B7D1',  # Sky Blue        
        '#F39C12',  # Amber
    ]
 
    ax = plt.subplot(2, 2, i_index+1)
    
    mark_every = len(df_max.columns) // 15
    
    for i in range(len(df_max)):
        ax.plot(df_max.iloc[i],
                marker=markers[i],
                markersize=3,
                linewidth=1.5,
                # linestyle=line_styles[i],
                label=df_max.index[i],
                color=colors[i],
                markevery=mark_every,
                
                )
    plt.title(f'{EV_num} EVSE', fontsize=12)
    
    
    if EV_num == 25:
        plt.ylim(-80_000, -18_000)
        plt.xlim(-100, 4000)
        plt.yticks([-7e4, -5.5e4, -4e4, -2.5e4])
    elif EV_num == 100:
        plt.ylim(-800_000, -80_000)
        plt.xlim(-100, 2100)
    elif EV_num == 500:
        plt.ylim(-40_000_000, -500_000)
        plt.xlim(-50, 1200)
        plt.yticks([-4e7, -2.5e7, -1e7])
    elif EV_num == 1000:
        plt.ylim(-200_000_000, -2_000_000)
        plt.xlim(-20, 410)
        
    
    # if i_index == 2:
    #     plt.legend(title='', loc='upper left', ncol=6,
    #                  shadow=True,
    #                  fontsize=12,
    #                  bbox_to_anchor=(-0.3, -0.3))
    
    # put legend under the plot
    if i_index == 2:
        plt.legend(title='', loc='upper center', ncol=3,
                        shadow=True,
                        fontsize=12,
                        bbox_to_anchor=(1, -0.5))
    

    import matplotlib.ticker as ticker

    # change the y ticks to scientific notation
    ax.ticklabel_format(axis='y',
                         style='sci',
                         scilimits=(5, 6),
                         useMathText=True)
    
    
    
    ax.yaxis.get_offset_text().set_x(-0.1)  # Move the offset text to the left
    ax.yaxis.get_offset_text().set_y(0.95)  # Adjust the y position if needed
    
    plt.yticks(fontsize=11)

    # plt.title('Best Eval. Model')
    plt.xlabel('Epochs')
    
    if i_index == 0 or i_index == 2:
        plt.ylabel('Reward' , fontsize=14)
    else:
        plt.ylabel('')

    # if evse >= 500:
    #     plt.xticks([0, 250, 500, 750, 1000], fontsize=12)
    # else:
    #     plt.xticks([0, 500, 1000, 1500, 2000], fontsize=12)

    # add grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(title='Number of EVSE', loc='lower right', ncol=2,
    #         shadow=True,
    #         fontsize=6)

    # if evse < 500:
    #     # Make the zoom-in plot:
    #     axins = zoomed_inset_axes(ax,
    #                             2.8,
    #                             loc="lower right")

    #     start_epoch_zoom = 1500
    #     for i in range(len(df_max)):
    #         axins.plot(df_max.iloc[i][start_epoch_zoom:],
    #                 marker=markers[i],
    #                 #    markersize=3,
    #                 linewidth=1.5,
    #                 linestyle=line_styles[i],
    #                 color=colors[i],
    #                 markevery=100
    #                 )

    #     axins.set_xlim(start_epoch_zoom, epoch_limit)
    #     axins.set_ylim(zoom_in_lower, zoom_in_upper)

    #     plt.xticks(visible=False)
    #     plt.yticks(fontsize=10)
    #     mark_inset(ax, axins,
    #             loc1=1,
    #             loc2=2,
    #             fc="0.92",
    #             ec="0.5",
    #             lw=1)

    #     # add grid
    #     plt.grid(True, which='Major', linestyle='--', linewidth=0.5)

    #     # plt.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
    #     plt.yticks(fontsize=12)

    #     plt.draw()

    
    continue
##################################
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
    
##################################
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

    # plt.ylim(upper_limit, lower_limit)
    # plt.xlim(-50, epoch_limit)

    plt.xlabel('Epochs')
    # remove the y_tick labels but keep the grid
    y_ticks = plt.gca().yaxis.get_major_ticks()
    for t in y_ticks:
        t.label1.set_visible(False)
        
    # if evse >= 500:
    #     plt.xticks([0, 250, 500, 750, 1000], fontsize=12)
    # else:
    #     plt.xticks([0, 500, 1000, 1500, 2000], fontsize=12)

    # add grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        
        
    
plt.tight_layout()
# set the bottom and top of the plot
plt.subplots_adjust(bottom=0.37, top=0.9, left=0.06, right=0.98)
plt.savefig('./Results_Analysis/SAC_TD3_max_rewards.png')
plt.show()
exit()