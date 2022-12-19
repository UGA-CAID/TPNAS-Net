import os
import numpy as np
import torch
import scipy.io as scio

from models import TPNAS_Net
from utils import squeeze_weights

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def visualize_results(results, mode='loss_acc', config=None):
    """
    Visualize train-val results, including loss and accuracy curves, testing
    confusion matrix and classification reports (only print).
    """

    if mode == 'loss_acc':
        
        # Plot the dataframes
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        axes[0].set_facecolor('w')
        axes[1].set_facecolor('w')
        
        # set axis color
        axes[0].spines['left'].set_color('black')
        axes[0].spines['bottom'].set_color('black')
        # axes[0].spines['right'].set_color('black')
        # axes[0].spines['top'].set_color('black')
        
        axes[1].spines['left'].set_color('black')
        axes[1].spines['bottom'].set_color('black')
        # axes[1].spines['right'].set_color('black')
        # axes[1].spines['top'].set_color('black')
        
        if isinstance(results[0], dict):
            # Create dataframes
            loss_df = pd.DataFrame.from_dict(results[0]).reset_index().melt(
                id_vars=['index']).rename(columns={"index": "Epochs",
                                                   "value": "Loss",
                                                   "variable": "Mode"})

            acc_df = pd.DataFrame.from_dict(results[1]).reset_index().melt(
                id_vars=['index']).rename(columns={"index": "Epochs",
                                                   "value": "Acc",
                                                   "variable": "Mode"})
            
            # Plot train/val loss/acc
            sns.lineplot(x="Epochs", 
                         y="Loss", 
                         data=loss_df, 
                         hue='Mode', 
                         ax=axes[0]).set_title('Training/Val Loss/Epoch')
    
            sns.lineplot(x="Epochs", 
                         y="Acc",
                         data=acc_df, 
                         hue='Mode', 
                         ax=axes[1]).set_title('Training/Val Accuracy/Epoch')

        else:
            loss_df = pd.DataFrame(results[0],
                                   columns={"Loss"}).reset_index(
            ).rename(columns={"index": "Epochs"})

            acc_df = pd.DataFrame(results[1],
                                  columns={"Acc"}).reset_index(
            ).rename(columns={"index": "Epochs"})
            
            # Plot
            sns.lineplot(x="Epochs", 
                         y="Loss", 
                         data=loss_df, 
                         ax=axes[0]).set_title('Training Loss/Epoch')
    
            sns.lineplot(x="Epochs", 
                         y="Acc",
                         data=acc_df, 
                         ax=axes[1]).set_title('Training Accuracy/Epoch')
            
        # Note: Do not use sns.set otherwise it will cause color change.
        # seaborn.set updates the colors if its color_codes argument 
        # is set to True (which is the default). if using sns.set, 
        # its color_codes argument must be set to False
        sns.set(font_scale=1.3, style='white', color_codes=False)
        
        # plt.draw()
        # plt.show()

        # Save fig
        plt.savefig(os.path.join(config.outdir, 'curve_loss_acc.png'), 
                    dpi=config.dpi)
        plt.close()

    elif mode == 'confusion':
        class_name = config.class_names  # dict: str
        if isinstance(results, dict):
            # Create dataframes
            train_df = pd.DataFrame(results['train']).rename(columns=class_name,
                                                             index=class_name)

            val_df = pd.DataFrame(results['val']).rename(columns=class_name,
                                                         index=class_name)

            # Plot the dataframes
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
            axes[0].set_facecolor('w')
            axes[1].set_facecolor('w')

            # Plot
            sns.heatmap(train_df/np.sum(train_df), # train_df or train_df/np.sum(train_df)
                        annot=True,
                        ax=axes[0],
                        fmt='.2%', # 'd' or '.2%'
                        cbar=False, 
                        cmap='Blues').set_title('Training Confusion Matrix')
            axes[0].set(xlabel="Predicted",
                        ylabel="True/Actual")

            sns.heatmap(val_df/np.sum(val_df), # val_df or val_df/np.sum(val_df)
                        annot=True,
                        ax=axes[1],
                        fmt='.2%', # 'd' or '.2%'
                        cbar=False, 
                        cmap='Blues').set_title('Val Confusion Matrix')
            axes[1].set(xlabel="Predicted",
                        ylabel="True/Actual")

        else:
            # Create dataframes
            df = pd.DataFrame(results).rename(columns=class_name,
                                              index=class_name)

            # fig = plt.figure(figsize=(10, 10))
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            ax.set_facecolor('w')
            ax = sns.heatmap(df/np.sum(df), 
                            annot=True, 
                            fmt='.2%', # 'd' or '.2%'
                            cbar=False, 
                            cmap='Blues', 
                            annot_kws={'fontsize': 14}, 
                            ax=ax, 
                            )
            # add frames
            for _, spine in ax.spines.items():
                spine.set_visible(True)
            
            plt.title('Confusion Matrix', fontsize=18)
            plt.xlabel("Predicted", fontsize=14)
            plt.ylabel("True/Actual", fontsize=14)
            
        # Note: don't use sns.set otherwise it will cause color change.
        # seaborn.set updates the colors if its color_codes argument 
        # is set to True (which is the default). if must use sns.set, 
        # its color_codes argument must be set to False
        sns.set(font_scale=1.3, style='white', color_codes=False)

        # plt.draw()
        # plt.show()
        
        # Save fig
        plt.savefig(os.path.join(config.outdir, 'confusion_matrix.png'), 
                    dpi=config.dpi)
        plt.close()

    elif mode == 'report':
        print('Classification reports:\n')

        if type(results) == dict:
            print('Training:\n')
            print(results['train'])

            print('Val:\n')
            print(results['val'])
        else:
            print(results)


def visualize_weights_filters(net=None, dg='all', config=None):
    """
    Visualize the filters in convolutional layer according to the sign of
    softmax (or last linear layer) weights.
    """

    if net is None:
        if config is None:
            print('Please specify config file in order to reload model')
            return
        
        if config.model == 'TPNAS-Net':
            net = TPNAS_Net(model='proxyless_gpu', pretrained=False) # model name can be self-defined accordingly
            
            # input channels of the first conv layer must match that of input data
            if not net.first_conv.conv.in_channels == config.in_feature:
                net.first_conv.conv = squeeze_weights(
                    net.first_conv.conv, 
                    mode='input', 
                    input_channels=config.in_feature)
            
            # output channels of the linear layer must match the classes
            if not net.classifier.linear.out_features == config.num_classes:
                net.classifier.linear = squeeze_weights(
                    net.classifier.linear, 
                    mode='output', 
                    output_features=config.num_classes)
        
            checkpoint = torch.load(os.path.join(config.outdir, 'trained_model.pth'))
            net.load_state_dict(checkpoint)

    # extract weights and conv filters
    if config.model == 'TPNAS-Net':
        # softmax or last linear layer weights
        weights = net.classifier.linear.weight.cpu().detach().numpy().transpose()
        
        # conv layer filters
        filters = net.feature_mix_layer.conv.weight.squeeze(dim=2)
        filters = filters.cpu().detach().numpy()
        
        # save filter type index
        draw_lineplot(config, weights, filters, flag=False)
        
    # save filters
    savename = os.path.join(config.outdir, 'filter_timeseries.mat')
    # f = h5py.File(savename, 'w')
    # f.create_dataset('filters', data=filters)
    # f.close()
    scio.savemat(savename, {'filter_timeseries': filters})

def draw_lollipop(config, weights):
    """
    Draw lollipop chart for classes(sulcal, 2HGs and 3HGs) weights 
    in the softmax layer.
    """
    
    # number of classes
    classes = config.classes
    class_name = [c+' Weights' for c in classes]

    wt_df = pd.DataFrame(weights,
                         columns=class_name)
    wt_df = wt_df.reset_index().rename(columns={"index": "Filter Index"})

    # index start from 1
    wt_df["Filter Index"] += 1

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('w')

    if len(classes) == 3:
        vmax1 = np.max(wt_df[class_name[0]])
        vmax2 = np.max(wt_df[class_name[1]])
        vmax3 = np.max(wt_df[class_name[2]])
        vmax = np.round(np.max([vmax1, vmax2, vmax3]), 1)

        # scatters
        sc_size = 32
        ax1.scatter(x=wt_df["Filter Index"],
                    y=wt_df[class_name[0]],
                    c='b',
                    marker='^',
                    s=sc_size,
                    )
        ax1.scatter(x=wt_df["Filter Index"],
                    y=wt_df[class_name[1]],
                    c='g',
                    marker='*',
                    s=sc_size,
                    )
        ax1.scatter(x=wt_df["Filter Index"],
                    y=wt_df[class_name[2]],
                    c='r',
                    marker='o',
                    s=sc_size,
                    )

        # vlines
        lw = 1.2
        ax1.vlines(x=wt_df["Filter Index"],
                   ymin=0,
                   ymax=wt_df[class_name[0]],
                   color='b', 
                   linestyle='solid', 
                   linewidth=lw,
                   )
        ax1.vlines(x=wt_df["Filter Index"],
                   ymin=0,
                   ymax=wt_df[class_name[1]],
                   color='g', 
                   linestyle='dashed', 
                   linewidth=lw,
                   )
        ax1.vlines(x=wt_df["Filter Index"],
                   ymin=0,
                   ymax=wt_df[class_name[2]],
                   color='r', 
                   linestyle='dashdot', 
                   linewidth=lw,
                   )

    elif len(classes) == 2:
        vmax1 = np.max(wt_df[class_name[0]])
        vmax2 = np.max(wt_df[class_name[1]])
        vmax = np.round(np.max([vmax1, vmax2]), 1)
        
        # scatters
        sc_size = 32
        ax1.scatter(x=wt_df["Filter Index"],
                    y=wt_df[class_name[0]],
                    c='b',
                    marker='^',
                    s=sc_size,
                    )
        ax1.scatter(x=wt_df["Filter Index"],
                    y=wt_df[class_name[1]],
                    c='r',
                    marker='*',
                    s=sc_size,
                    )

        # vlines
        lw = 1.2
        ax1.vlines(x=wt_df["Filter Index"],
                   ymin=0,
                   ymax=wt_df[class_name[0]],
                   color='b', 
                   linestyle='solid', 
                   linewidth=lw,
                   )
        ax1.vlines(x=wt_df["Filter Index"],
                   ymin=0,
                   ymax=wt_df[class_name[1]],
                   color='r', 
                   linestyle='dashdot', 
                   linewidth=lw,
                   )

    # hide right and top axis
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')

    # set left and bottom axis color and linewidth
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_linewidth(3)
    ax1.spines['bottom'].set_linewidth(3)

    # set ticks and lengend
    plt.yticks([-vmax, 0, vmax], fontsize=16, fontweight='bold')
    xmax = len(weights)
    if xmax > 200:
        xinterval = int(len(weights)/50)*10
    else:
        xinterval = 10

    plt.xticks(range(0, xmax, xinterval), fontsize=16, fontweight='bold')

    legend_properties = {'weight': 'bold', 'size': 12}
    plt.legend(class_name,
               bbox_to_anchor=(0.005, 1.1),
               loc="upper left",
               prop=legend_properties)

    # plot horizontal (zero) axis
    plt.axhline(y=0, color='k', linestyle='-', linewidth=1.5)

    # plt.draw()
    # plt.show()
    
    # Save fig
    plt.savefig(os.path.join(config.outdir, 'weights_softmax.png'), 
                dpi=config.dpi)
    plt.close()


def draw_lineplot(config, weights, filters, flag=True):
    """
    Draw subplots of lineplot for all the filters in the conv layer.
    """
    
    # classes names
    classes = config.classes

    if len(classes) == 3:
        # determine the row corresponding to sulc, 2HGs and 3HGs, respectively
        idx_sulc = []
        idx_2HGs = []
        idx_3HGs = []
        idx_other = []
        for i, w in enumerate(weights):
            if w[0] > 0 and w[1] < 0 and w[2] < 0:
                idx_sulc.append(i)
            elif w[1] > 0 and w[0] < 0 and w[2] < 0:
                idx_2HGs.append(i)
            elif w[2] > 0 and w[0] < 0 and w[1] < 0:
                idx_3HGs.append(i)
            else:
                idx_other.append(i)

        # divide filters based on the sulc, 2HGs and 3HGs idx
        filt_sulc = filters[idx_sulc]
        filt_2HGs = filters[idx_2HGs]
        filt_3HGs = filters[idx_3HGs]
        filt_other = filters[idx_other]

        # save filter tpye index
        savename = os.path.join(config.outdir, 'filter_index.mat')
        # f = h5py.File(savename, 'w')
        # f.create_dataset('filter_sulcal_index', data=np.array(idx_sulc)+1)
        # f.create_dataset('filter_2HGs_index', data=np.array(idx_2HGs)+1)
        # f.create_dataset('filter_3HGs_index', data=np.array(idx_3HGs)+1)
        # f.close()
        scio.savemat(savename, {'filter_sulcal_index': np.array(idx_sulc)+1,
                                'filter_2HGs_index': np.array(idx_2HGs)+1,
                                'filter_3HGs_index': np.array(idx_3HGs)+1})

        if flag:
            filters_sorted = np.concatenate((filt_sulc,
                                             filt_2HGs,
                                             filt_3HGs,
                                             filt_other), axis=0)
    
            if len(filters_sorted) > 128:
                nrows = int(np.ceil((len(filters_sorted) - len(filt_other)) / 8))
            else:
                nrows = int(len(weights) / 8)
            ncols = 8
            fig = plt.figure(figsize=(12, 12))
            for nrow in range(1, nrows+1):
                for ncol in range(1, ncols+1):
                    idx = (nrow-1) * 8 + ncol
                    ax = fig.add_subplot(nrows, ncols, idx)
                    ax.set_facecolor('w')
                
                    # set axis color
                    ax.spines['left'].set_color('black')
                    ax.spines['bottom'].set_color('black')
                    ax.spines['right'].set_color('black')
                    ax.spines['top'].set_color('black')
    
                    df = pd.DataFrame(filters_sorted[(nrow-1) * 8 + (ncol-1)],
                                      columns={'Amplitude'}).reset_index(
                    ).rename(columns={'index': 'Time'})
    
                    if idx < len(filt_sulc) + 1:
                        g2 = sns.lineplot(x='Time',
                                          y='Amplitude',
                                          data=df,
                                          ax=ax,
                                          color='b')  # sulc
                    elif idx < len(filt_sulc) + len(filt_2HGs) + 1:
                        g2 = sns.lineplot(x='Time',
                                          y='Amplitude',
                                          data=df,
                                          ax=ax,
                                          color='g')  # 2HGs
                    elif idx < len(filters_sorted) - len(filt_other) + 1:
                        g2 = sns.lineplot(x='Time',
                                          y='Amplitude',
                                          data=df,
                                          ax=ax,
                                          color='r')  # 3HGs
                    else:
                        g2 = sns.lineplot(x='Time',
                                          y='Amplitude',
                                          data=df,
                                          ax=ax,
                                          color='k')  # others

                    # g2.axis("off")
                    # if idx == 25:
                    #     g2.set_ylabel("Amplitude")
                    #     g2.set_yticks([])
                    #     g2.get_xaxis().set_visible(False)
                    # elif idx == 60:
                    #     g2.set_xlabel("Time (TR)")
                    #     g2.set_xticks([])
                    #     g2.get_yaxis().set_visible(False)
                    # else:
                    #     g2.get_xaxis().set_visible(False)
                    #     g2.get_yaxis().set_visible(False)
    
                    # turn off the frame of each subplot
                    # ax.axis("off")
    
                    # turn off the axis ticks of each subplot
                    g2.get_xaxis().set_visible(False)
                    g2.get_yaxis().set_visible(False)

    elif len(classes) == 2:
        idx_1 = []
        idx_2 = []
        idx_other = []
        for i, w in enumerate(weights):
            if w[0] > 0 and w[1] < 0:
                idx_1.append(i)
            elif w[1] > 0 and w[0] < 0:
                idx_2.append(i)
            else:
                idx_other.append(i)

        # divide filters based on class 1 and 2 idx
        filt_1 = filters[idx_1]
        filt_2 = filters[idx_2]
        filt_other = filters[idx_other]
        
        # save filter tpye index
        if 'Sulcal' in config.classes:
            savename = os.path.join(config.outdir, 'filter_index.mat')
            # f = h5py.File(savename, 'w')
            # f.create_dataset('filter_sulcal_index', data=np.array(idx_1) + 1)
            # f.create_dataset('filter_gyral_index', data=np.array(idx_2) + 1)
            # f.close()
            scio.savemat(savename, {'filter_sulcal_index': np.array(idx_1)+1,
                                    'filter_gyral_index': np.array(idx_2)+1})
            
        elif '2HGs' in config.classes:
            savename = os.path.join(config.outdir, 'filter_index.mat')
            # f = h5py.File(savename, 'w')
            # f.create_dataset('filter_2HGs_index', data=np.array(idx_1) + 1)
            # f.create_dataset('filter_3HGs_index', data=np.array(idx_2) + 1)
            # f.close()
            scio.savemat(savename, {'filter_2HGs_index': np.array(idx_1)+1,
                                    'filter_3HGs_index': np.array(idx_2)+1})

        if flag:
            filters_sorted = np.concatenate((filt_1,
                                             filt_2,
                                             filt_other), axis=0)
    
            if len(filters_sorted) > 128:
                nrows = int(np.ceil((len(filters_sorted) - len(filt_other)) / 8))
            else:
                nrows = int(len(weights) / 8)
            ncols = 8
            fig = plt.figure(figsize=(12, 12))
            for nrow in range(1, nrows+1):
                for ncol in range(1, ncols+1):
                    idx = (nrow-1) * 8 + ncol
                    ax = fig.add_subplot(nrows, ncols, idx)
                    ax.set_facecolor('w')
                    
                    # set axis color
                    ax.spines['left'].set_color('black')
                    ax.spines['bottom'].set_color('black')
                    ax.spines['right'].set_color('black')
                    ax.spines['top'].set_color('black')

                    df = pd.DataFrame(filters_sorted[(nrow-1) * 8 + (ncol-1)],
                                      columns={'Amplitude'}).reset_index(
                    ).rename(columns={'index': 'Time'})
    
                    if idx < len(filt_1) + 1:
                        g2 = sns.lineplot(x='Time',
                                          y='Amplitude',
                                          data=df,
                                          ax=ax,
                                          color='b')  # sulc or 2HGs
                    elif idx < len(filt_1) + len(filt_2) + 1:
                        g2 = sns.lineplot(x='Time',
                                          y='Amplitude',
                                          data=df,
                                          ax=ax,
                                          color='r')  # 2HGs or 3HGs
                    else:
                        g2 = sns.lineplot(x='Time',
                                          y='Amplitude',
                                          data=df,
                                          ax=ax,
                                          color='k')  # others
    
                    # turn off the frame of each subplot
                    # ax.axis("off")
    
                    # turn off the axis ticks of each subplot
                    g2.get_xaxis().set_visible(False)
                    g2.get_yaxis().set_visible(False)

    if flag:
        fig.text(0.5, 0.08, 'Time (TR)', ha='center', fontsize=20)
        fig.text(0.08, 0.5, 'Amplitude', va='center', fontsize=20,
                 rotation='vertical')
    
        # plt.draw()
        # plt.show()
    
        # Save fig
        plt.savefig(os.path.join(config.outdir, 'filters.png'), 
                    dpi=config.dpi)
        plt.close()










