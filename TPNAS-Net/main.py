import os
import torch.utils.data as D
import torch

from trainer import Trainer
from dataloader import CortexDataset, train_val_dataset
from config import get_config

from utils import idx2class, read_and_save_vtk
from plot_utils import visualize_results


def main(config):
    # load data
    dataset = CortexDataset(rootdir=config.datadir, tasks=config.tasks)
        
    # split dataset into train and val = 7:3
    datasets = train_val_dataset(config=config, dataset=dataset, val_split=0.3)
        
    # data loader
    data_loader = {}
    data_loader['train'] = D.DataLoader(datasets['train'], 
                                        batch_size=config.batch_size, 
                                        shuffle=config.shuffle, 
                                        num_workers=config.workers, 
                                        pin_memory=True,
                                        )
    
    data_loader['val'] = D.DataLoader(datasets['val'], 
                                      batch_size=config.batch_size, 
                                      # shuffle=config.shuffle, 
                                      num_workers=config.workers, 
                                      pin_memory=True, 
                                      )

    # convert one-hot encoding of labels to label names
    if config.tasks == 'hinges':
        config.classes = ['2HGs', '3HGs']
    elif config.tasks == 'sulc_gyral':
        config.classes = ['Sulcal', 'Gyral']
    else:
        config.classes = ['Sulcal', '2HGs', '3HGs']

    # class name
    labels = dataset.label.squeeze(1).tolist()
    class_names = idx2class(labels, classes=config.classes)
    config.class_names = class_names[1]
    
    # input and output nums
    config.in_feature = dataset.datadim
    config.num_classes = dataset.classes
    
    # input data length: input_size
    config.input_size = dataset.input_size

    # optimizer, default: SGD
    # config.opt_name = 'Adam'
    config.opt_name = 'SGD'
    
    # loss function, default: CE (Cross-Entropy)
    config.crit_name = 'CE'
    
    # initialize trainer
    trainer = Trainer(config, data_loader)

    # train and val
    results = trainer.train_val_model()

    # Visualize loss and accuracy
    visualize_results(results=results[1],
                      mode='loss_acc',
                      config=config)

    # Visualize confusion matrix
    visualize_results(results=results[2][0],
                      mode='confusion',
                      config=config)

    # Visualize classification reports
    visualize_results(results=results[2][1],
                      mode='report')

    # # Save labels and preds mapped into the surface
    # read_and_save_vtk(data=results[3],
    #                   config=config)


if __name__ == '__main__':
    # get basic configurations
    config, unparsed = get_config()
    print(config)

    # cpu or gpu
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # subjects
    subjects = ['sub1', 'sub2']
    for subject in subjects:
        datadir = os.path.join(config.root_dir, subject)
        config.datadir = datadir
        config.outdir = os.path.join(datadir, config.tasks)  # out dir
        
        main(config=config)
    
    
    
    
    
    






















