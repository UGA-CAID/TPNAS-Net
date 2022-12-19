import os
import numpy as np
import nighres as ng

from config import get_config


def count_parameters(model):
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    return total_params


def idx2class(labels: list, classes=None):
    """
    Convert one-shot encoders of labels to original labels.
    """
    
    idx = list(set(labels))
    
    if len(idx) == 3:
        classes = ['Sulc', '2HGs', '3HGs']
        res = {k: c for k, c in enumerate(classes)}
        res_all = [classes[c] for c in labels]
    elif len(idx) == 2:
        if classes is None:
            print("Please specify class labels. e.g., ['Sulc', 'Gyri']")
            return
        else:
            res = {k: c for k, c in enumerate(classes)}
            res_all = [classes[c] for c in labels]
    
    return res_all, res


def squeeze_weights(net, mode='input', input_channels=1, output_features=2):
    '''
        Adjust the initialized weights to match the number of input data 
        channels (mode='input') or output class labels (mode='output').
    '''

    if mode == 'input':
        net.weight.data = net.weight.data.sum(dim=1)[:, None]
        net.in_channels = input_channels
    elif mode == 'output':
        temp_w = net.weight.data[0 : output_features, :]
        net.weight.data = temp_w
        
        temp_b = net.bias.data[0 : output_features]
        net.bias.data = temp_b
        
        net.out_features = output_features
    else:
        raise Exception('mode is either input or output.')

    return net


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def read_and_save_vtk(data, config, fname=None):
    """
    Save as a VTK file. Take data (a tuple including idx_surfs, y_labels, 
    y_preds) as input; surf_dir (or datadir) is the directory to read brain surface vtk
    file; config is the configuration file. fname: default is None, using 
    default saving filename, e.g., 'labels', 'preds'. if not, it will use
    'gap' as filename. In addition, 
    """
    
    hemi = ['lh', 'rh']
    for h in hemi:
        surfdir_h = os.path.join(config.datadir, h + '.white_new.vtk')
                
        # read surf vtk file
        surf_dict = ng.io.load_mesh(surfdir_h)
        
        # save file name
        if fname is None:
            suffix_label = h + '_labels'
            suffix_pred = h + '_preds'
        else:
            suffix_label = h + '_' + fname
        
        # include train and test mode
        if isinstance(data[0], dict):
            for mode in data[0].keys():
                idx_surfs = data[0][mode]
                y_labels = data[1][mode]
                
                # discriminate left and right hemisphere index
                if h == 'lh':
                    ind = [idx for idx, val in enumerate(idx_surfs) 
                           if val < 1000000]
                else:
                    ind = [idx for idx, val in enumerate(idx_surfs) 
                           if val > 1000000]
                
                y_labels_h = []
                idx_surfs_h = []
                for i in ind:
                    y_labels_h.append(y_labels[i])
                    
                    if h == 'lh':
                        idx_surfs_h.append(idx_surfs[i])
                    else:
                        idx_surfs_h.append(idx_surfs[i] - 1000000)
                
                # save labels
                labels_all = -np.ones(surf_dict['points'].shape[0], 
                                      dtype=np.int32)
                labels_all[idx_surfs_h] = y_labels_h
                surf_dict['data'] = labels_all
                fname_labels = os.path.join(config.out_dir, 
                                            suffix_label + '_' + mode + '.vtk')
                ng.io.save_mesh(fname_labels, surf_dict)
                del surf_dict['data']
                
                # include idx_surf, y_labels and y_preds
                if len(data) == 3:
                    y_preds = data[2][mode]
                    y_preds_h = []
                    for i in ind:
                        y_preds_h.append(y_preds[i])
                    
                    # save preds
                    preds_all = -np.ones(surf_dict['points'].shape[0], 
                                         dtype=np.int32)
                    preds_all[idx_surfs_h] = y_preds_h
                    surf_dict['data'] = preds_all
                    fname_preds = os.path.join(config.out_dir, 
                                               suffix_pred + '_' + mode + '.vtk')
                    ng.io.save_mesh(fname_preds, surf_dict)
                    del surf_dict['data']
                    
        else:
            idx_surfs = data[0]
            y_labels = data[1]
            
            # discriminate left and right hemisphere index
            if h == 'lh':
                ind = [idx for idx, val in enumerate(idx_surfs) 
                       if val < 1000000]
            else:
                ind = [idx for idx, val in enumerate(idx_surfs) 
                       if val > 1000000]
            
            y_labels_h = []
            idx_surfs_h = []
            for i in ind:
                y_labels_h.append(y_labels[i])
                
                if h == 'lh':
                    idx_surfs_h.append(idx_surfs[i])
                else:
                    idx_surfs_h.append(idx_surfs[i] - 1000000)
                
            # save labels
            labels_all = -np.ones(surf_dict['points'].shape[0], dtype=np.int32)
            labels_all[idx_surfs_h] = y_labels_h
            surf_dict['data'] = labels_all
            fname_labels = os.path.join(config.out_dir, suffix_label + '.vtk')
            ng.io.save_mesh(fname_labels, surf_dict)
            del surf_dict['data']
            
            # include idx_surf, y_labels and y_preds
            if len(data) == 3:
                y_preds = data[2]
                y_preds_h = []
                for i in ind:
                    y_preds_h.append(y_preds[i])
            
                # save preds
                preds_all = -np.ones(surf_dict['points'].shape[0], dtype=np.int32)
                preds_all[idx_surfs_h] = y_preds_h
                surf_dict['data'] = preds_all
                fname_preds = os.path.join(config.out_dir, suffix_pred + '.vtk')
                ng.io.save_mesh(fname_preds, surf_dict)
                del surf_dict['data']


# if __name__ == '__main__':
#     # get basic configurations
#     config, unparsed = get_config()
#     print(config)

#     # subjects
#     subjects = ['sub1', 'sub2']
#     for subject in subjects:
#         datadir = os.path.join(config.datadir, subject)
#         config.datadir = datadir
#         config.outdir = datadir  # out dir

#         read_and_save_vtk(data=None, config=config)














