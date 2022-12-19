import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models import TPNAS_Net
from utils import count_parameters, squeeze_weights
from utils import AverageMeter

from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for training the model.
    """
    
    def __init__(self, config, data_loader):
        
        # data loader
        self.data_loader = data_loader
        
        # cpu or gpu
        self.device = config.device
        
        # set seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            cudnn.benchmark = True
        
        # training params
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.init_lr = config.init_lr
        self.weight_decay = config.weight_decay
        
        # model params
        self.model_name = config.model
        self.in_feature = config.in_feature
        self.num_classes = config.num_classes
        
        # data info
        self.input_size = config.input_size
        
        # model parameter adjustment
        if self.model_name == 'TPNAS-Net':
            self.model = TPNAS_Net(model='proxyless_gpu', pretrained=False)
            self.model = self.model.double().to(self.device)
        
        # input channels of the first conv layer must match that of input data
        if not self.model.first_conv.conv.in_channels == self.in_feature:
            self.model.first_conv.conv = squeeze_weights(
                self.model.first_conv.conv, 
                mode='input', 
                input_channels=self.in_feature)
        
        # output channels of the linear layer must match the classes
        if not self.model.classifier.linear.out_features == self.num_classes:
            self.model.classifier.linear = squeeze_weights(
                self.model.classifier.linear, 
                mode='output', 
                output_features=self.num_classes)
            
        
        # out dir
        self.outdir = config.outdir
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        
        # loss function
        self.crit_name = config.crit_name
        if config.crit_name == 'CE':
            self.criterion = nn.CrossEntropyLoss()
            self.criterion = self.criterion.to(self.device)
        
        # optimizer
        self.opt_name = config.opt_name
        if config.opt_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), 
                                        lr=self.init_lr, 
                                        weight_decay=self.weight_decay)
        elif config.opt_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), 
                                       lr=self.init_lr, 
                                       weight_decay=self.weight_decay)
        
        # LR scheduler
        if config.lr_scheduler == 'StepLR':
            self.scheduler_step = 5
        else:
            self.scheduler_step = None
    
    
    def adjust_learning_rate(self, epoch):
        """
        Set the learning rate to the initial LR decayed by 5 
        every 'self.scheduler_step' epochs.
        """
        if self.scheduler_step is not None:
            lr = self.init_lr * (0.1 ** (epoch // self.scheduler_step))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    
    def train_one_epoch(self, epoch):
        """
        Train a model in one epoch, which is only used in train_model().
        """
        
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        idx_surfs = []
        y_labels = []
        y_preds = []
        
        # switch to train mode
        self.model.train()
        
        if isinstance(self.data_loader, dict):
            train_dataloader = self.data_loader['train']
        else:
            train_dataloader = self.data_loader
        
        # iterate over data.
        for inputs, labels in train_dataloader:
            idx_surf = inputs[:, :, 0].cpu().numpy().astype(np.int64)
            idx_surf = idx_surf.squeeze(1)
                
            inputs = inputs[:, :, 1:]
            inputs = inputs.to(self.device)
            labels = labels.squeeze(1)
            labels = labels.to(self.device)
            
            logits, _ = self.model(inputs)  # out, gap_embed
            _, preds = torch.max(logits, 1)
            
            loss = self.criterion(logits, labels)
            acc = accuracy_score(labels.detach().cpu(), preds.detach().cpu())
            
            train_loss.update(loss.item(), inputs.size(0))
            train_acc.update(acc, inputs.size(0))
            
            # compute gradients and update optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if epoch == self.epochs:
                idx_surfs += idx_surf.tolist()
                y_labels += labels.cpu().numpy().tolist()
                y_preds += preds.cpu().numpy().tolist()

        res1 = (train_loss, train_acc)
        res2 = (idx_surfs, y_labels, y_preds)
        
        return res1, res2
    
    def train_val_model(self):
        """
        Trains and Valids the model.
        """
        
        # count net parameters
        total_params = count_parameters(self.model)
        
        # training log
        logdir = os.path.join(self.outdir, 'log_train_val.txt')
        logfile = open(logdir, 'wt')
        logfile.write('Model: {} \n'.format(self.model_name))
        print('Model: {} \n'.format(self.model_name))
        
        logfile.write('Total parameters: ' + str(total_params) + '\n')
        print('Total parameters: ' + str(total_params) + '\n')
        
        logfile.write('Loss Function: {} \n'.format(self.crit_name))
        print('Loss Function: {} \n'.format(self.crit_name))
        
        logfile.write('Optimizer: {} \n'.format(self.opt_name))
        print('Optimizer: {} \n'.format(self.opt_name))
        
        print('Training and val begin \n') # ---training begin---
        
        # training accuracy and loss
        loss_stats = {'train': [], 'val': []}
        acc_stats = {'train': [], 'val': []}
        
        idx_surfs_stats = {'train': [], 'val': []}
        y_labels_stats = {'train': [], 'val': []}
        y_preds_stats = {'train': [], 'val': []}
        
        # training along epoch
        for epoch in range(1, self.epochs+1):
            print('Epoch {}/{}'.format(epoch, self.epochs))
            print('-' * 20)
            
            # Adjust learning rate according to scheduler
            self.adjust_learning_rate(epoch)
            
            # train for one epoch
            res1_train, res2_train = self.train_one_epoch(epoch)
            
            # evaluate on validation set after training on each epoch
            res1_val, res2_val = self.val_model(epoch)
                                
            if epoch == self.epochs:
                idx_surfs_stats['train'] = res2_train[0]
                y_labels_stats['train'] = res2_train[1]
                y_preds_stats['train'] = res2_train[2]
                
                idx_surfs_stats['val'] = res2_val[0]
                y_labels_stats['val'] = res2_val[1]
                y_preds_stats['val'] = res2_val[2]
                
            loss_stats['train'].append(res1_train[0].avg)
            acc_stats['train'].append(res1_train[1].avg)

            loss_stats['val'].append(res1_val[0].avg)
            acc_stats['val'].append(res1_val[1].avg)
            
            msg = 'Train loss: {:.4f} - acc: {:.4f}\n'
            logfile.write(msg.format(res1_train[0].avg, 
                                     res1_train[1].avg))
            print(msg.format(res1_train[0].avg, 
                             res1_train[1].avg))
            
            msg = 'Val loss: {:.4f} - acc: {:.4f}\n'
            logfile.write(msg.format(res1_val[0].avg, 
                                     res1_val[1].avg))
            print(msg.format(res1_val[0].avg, 
                             res1_val[1].avg))
        
        print('Training and val end \n') # ---training end---
        
        # save the trained model in the last epoch
        fname_model = os.path.join(self.outdir, 'trained_model.pth')
        trained_model = self.model.state_dict()
        torch.save(trained_model, fname_model)
        
        # compute confusion matrix and classification report
        confusion_mx = {x: confusion_matrix(y_labels_stats[x], 
                                            y_preds_stats[x], 
                                            ) for x in ['train', 'val']}
        class_rep = {x: classification_report(y_labels_stats[x], 
                                              y_preds_stats[x], 
                                              ) for x in ['train', 'val']}
        
        logfile.write('\nConfusion matrix in train:\n')
        logfile.write('{}\n'.format(confusion_mx['train']))
        
        logfile.write('\nConfusion matrix in val:\n')
        logfile.write('{}\n'.format(confusion_mx['val']))
        
        logfile.write('\nClassification report in train:\n')
        logfile.write('{}\n'.format(class_rep['train']))
        
        logfile.write('\nClassification report in val:\n')
        logfile.write('{}\n'.format(class_rep['val']))
        
        logfile.close()
        
        print('\nConfusion matrix in train:\n')
        print('{}\n'.format(confusion_mx['train']))
        
        print('\nConfusion matrix in val:\n')
        print('{}\n'.format(confusion_mx['val']))
        
        print('\nClassification report in train:\n')
        print('{}\n'.format(class_rep['train']))
        
        print('\nClassification report in val:\n')
        print('{}\n'.format(class_rep['val']))
        
        res1 = (loss_stats, acc_stats)
        res2 = (confusion_mx, class_rep)
        res3 = (idx_surfs_stats, y_labels_stats, y_preds_stats)
        
        return self.model, res1, res2, res3

    def val_model(self, epoch=None):
        """
        Validate/test a model. If epoch is None, it only test a model, 
        otherwise validate a model for every opoch that is only used 
        in train_val_model().
        """
        
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        
        idx_surfs = []
        y_labels = []
        y_preds = []
        
        if isinstance(self.data_loader, dict):
            val_dataloader = self.data_loader['val']
        else:
            val_dataloader = self.data_loader
        
        with torch.no_grad():
            
            # switch to train mode
            self.model.eval()
            
            # iterate over data.
            for inputs, labels in val_dataloader:
                idx_surf = inputs[:, :, 0].cpu().numpy().astype(np.int64)
                idx_surf = idx_surf.squeeze()
                    
                inputs = inputs[:, :, 1:]
                inputs = inputs.to(self.device)
                labels = labels.squeeze()
                labels = labels.to(self.device)

                logits, _ = self.model(inputs) # out, gap_embed
                _, preds = torch.max(logits, 1)
                
                loss = self.criterion(logits, labels)
                acc = accuracy_score(labels.cpu(), preds.cpu())
                
                val_loss.update(loss.item(), inputs.size(0))
                val_acc.update(acc, inputs.size(0))
                
                if epoch is None or epoch == self.epochs:
                    idx_surfs += idx_surf.tolist()
                    y_labels += labels.cpu().numpy().tolist()
                    y_preds += preds.cpu().numpy().tolist()
        
        res1 = (val_loss, val_acc)
        res2 = (idx_surfs, y_labels, y_preds)
        
        return res1, res2






