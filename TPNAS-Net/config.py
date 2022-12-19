import argparse


parser = argparse.ArgumentParser(description='Basic config')

# data dir
parser.add_argument('--root_dir', type=str, default='./DATA',
                    help='data root directory')

# task: "three_classes", "sulc_gyral", and "hinges"
parser.add_argument('--tasks', type=str, default='three_classes', 
                    help='task: three classes of tasks: sulcal, 2HGs and 3HGs')

# seed
parser.add_argument('--seed', type=int, default=0, 
                    help='Random seed')

# training params
parser.add_argument('--batch_size', type=int, default=200, 
                    help='batch size')
parser.add_argument('--epochs', type=int, default=16, 
                    help='number of epochs for training')
parser.add_argument('--shuffle', type=bool, default=True, 
                    help='whether to shuffle the datasets')
parser.add_argument('--init_lr', type=float, default=0.01, 
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, 
                    help='weight decay rate')
parser.add_argument('--lr_scheduler', type=str, default='StepLR',
                    help='LR scheduler')
parser.add_argument('--workers', type=int, default=4, 
                    help="The batch on every device for training and val")

# model params
parser.add_argument('--model', type=str, default='TPNAS-Net', 
                    help='the model')
# parser.add_argument('--in_feature', type=int, default=1, 
#                     help='channels of input data')
# parser.add_argument('--num_classes', type=int, default=3, 
#                     help='number of classes')

# saving plot
parser.add_argument('--dpi', type=int, default=300, 
                    help='dpi of saving plot')

def get_config():
    config, unparsed = parser.parse_known_args()
    
    return config, unparsed












