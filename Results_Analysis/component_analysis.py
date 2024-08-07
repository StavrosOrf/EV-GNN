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

plt.figure(figsize=(6,4))
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = ['serif']
    
for i_index in range(2):
    i_index = 0
    # Load the data    
    
    if i_index == 0:
        data = pd.read_csv(f'./Results_Analysis/raw_data/SAC_No_Pruning_100.csv')
    else:
        data = pd.read_csv(f'./Results_Analysis/raw_data/TD3_No_Pruning_100.csv')
        
    # if i_index == 0:
    exp_name = 'Full Graph'
    # else:
    #     exp_name = 'No Postional Embedding'
        
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
    
    
    data.index = data.index.str.replace('SAC_ActionGNN', f'SAC EV-GNN\n({exp_name})')
    data.index = data.index.str.replace('SAC_GNN', f'SAC GNN-FX\n({exp_name})')
    data.index = data.index.str.replace('SAC_run', 'SAC')
    
    data.index = data.index.str.replace('TD3_ActionGNN', f'TD3 EV-GNN\n({exp_name})')
    data.index = data.index.str.replace('TD3_GNN', f'TD3 GNN-FX\n({exp_name})')
    data.index = data.index.str.replace('TD3_run', 'TD3')
    
    data_exp = data
    
    if i_index == 0:
        data = pd.read_csv(f'./Results_Analysis/raw_data/SAC_100.csv')
    else:
        data = pd.read_csv(f'./Results_Analysis/raw_data/TD3_100.csv')

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
    
    # rename the index to 'Algorithm'
    print(data)
    
    # if index contains 'SAC_ActionGNN' then rename to 'SAC'
    data.index = data.index.str.replace('SAC_ActionGNN', 'SAC EV-GNN')
    data.index = data.index.str.replace('SAC_GNN', 'SAC GNN-FX')
    data.index = data.index.str.replace('SAC_run', 'SAC')
    
    data.index = data.index.str.replace('TD3_ActionGNN', 'TD3 EV-GNN')
    data.index = data.index.str.replace('TD3_GNN', 'TD3 GNN-FX')
    data.index = data.index.str.replace('TD3_run', 'TD3')
    
    print(data_exp)

    df= pd.concat([data, data_exp], axis=0)
    
    #keep the first 1000 episodes
    df = df.iloc[:, :750]
    print(df)
    
    # fill the NaN values with the previous value
    # df = df.fillna(method='ffill', axis=1)

    # create a new dataframe with grouped rows based on the first 3 characters of the index
    df_mean = df.groupby(df.index).mean()


    # create df_max which contains the maximum value so far of each group

    df_max = df.groupby(df.index).cummax(axis=1)

    # print the maximum value of each group
    max_names = df_max.max(axis=1)

####################
    # df_mean = df_max.groupby(df_max.index).max().cummax(axis=1)
####################


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
    # df_max = df_max.reindex(['SAC', 'TD3',
    #                         'SAC GNN-FX', 'TD3 GNN-FX',
    #                         'SAC EV-GNN', 'TD3 EV-GNN'])


    colors = [
        '#16A085',   # Green Sea
        '#E74C3C',  # Bright Red
        '#2C3E50',  # Dark Blue (Slate)
        
        '#8E44AD',  # Wisteria Purple
        '#45B7D1',  # Sky Blue
        
        '#F39C12',  # Amber
    ]

    ax = plt.subplot(1, 1, 1)
    
    mark_every = len(df_max.columns) // 15
    

    # df_mean = df_mean.ewm(
    #     alpha=0.5,
    #     # span=10,
    #     # com=10,
    #     adjust=True).mean()
    
##################################
    # show variation of rewards over time

    # df_mean = df_mean.fillna(method='ffill', axis=1)

    # show the maximum value of each group and the minimum value of each group
    df_max = df.groupby(df.index).max()
    df_min = df.groupby(df.index).min()

    # plot every row
    for i in range(len(df_mean)):
        plt.plot(df_mean.iloc[i],
                label=df_mean.index[i],
                 markersize=3,
                linewidth=0.5,
                markevery=mark_every,
                marker=markers[i],
                color=colors[i])

        plt.fill_between(df_mean.columns,
                        df_min.iloc[i],
                        df_max.iloc[i],
                        color=colors[i],
                        alpha=0.2)

    # if EV_num == 25:
        
            # change the y ticks to scientific notation
    ax.ticklabel_format(axis='y',
                         style='sci',
                         scilimits=(5, 6),
                         useMathText=True)
    ax.yaxis.get_offset_text().set_x(-0.1)  # Move the offset text to the left
    ax.yaxis.get_offset_text().set_y(0.95)  # Adjust the y position if needed
    
    plt.yticks(fontsize=11)
    
    plt.legend(title='', loc='lower center', ncol=3,
               shadow=True,
               fontsize=11,
               bbox_to_anchor=(0.5, -0.5))
    plt.ylim(-2_580_000, -100_000)
    
    if i_index == 0:
        plt.xlim(-10, 650)
    else:
        plt.xlim(-10, 750)

    plt.xlabel('Epochs')
        
    plt.ylabel('Rewards')
    
    # remove the y_tick labels but keep the grid
    # y_ticks = plt.gca().yaxis.get_major_ticks()
    # for t in y_ticks:
    #     t.label1.set_visible(False)
        
    # if evse >= 500:
    #     plt.xticks([0, 250, 500, 750, 1000], fontsize=12)
    # else:
    #     plt.xticks([0, 500, 1000, 1500, 2000], fontsize=12)

    # add grid
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='', loc='lower center', ncol=3,
               shadow=True,
               fontsize=11,
               bbox_to_anchor=(0.5, -0.5))
        
        
    
    # plt.tight_layout()
    # set the bottom and top of the plot
    plt.subplots_adjust(bottom=0.3)
    plt.savefig('./Results_Analysis/Component_Analysis.png')
    plt.show()
    exit()