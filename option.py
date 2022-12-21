import argparse
from models.common import *

parser = argparse.ArgumentParser(description='An Efficient Unfolding Network for HSI SR')

# GPU setting
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--gpus', type=str, default="0, 1", help='gpu ids')
parser.add_argument('--n_threads', type=int, default=8, help='number of threads for data loading')
parser.add_argument("--seed", type=int, default=1, help="start seed for model")

# Dataset Setting
parser.add_argument('--scale', type=int, default=2, help='super resolution scale')
parser.add_argument('--n_colors', type=int, default=102, help='number of spectral channels to use')
parser.add_argument('--dir_data', type=str, default='/home/denghong/hdata/data/', help='dataset directory')
parser.add_argument('--dataset_name', type=str, default='Pavia', help='train dataset name')
parser.add_argument('--sigma', type=float, default=0., help='noise level')
parser.add_argument('--is_blur', type=bool, default=False, help='False for Bicubic downsampling '
                                                                'True for Gaussian downsampling')

# storage setting
parser.add_argument('--save_dir', type=str, default='hsr', help='results directory')

# training setting
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
parser.add_argument("--resume", default="", type=str, help="path to checkpoint")
parser.add_argument('--reset', action='store_true', help='reset the training')

# loss function
parser.add_argument('--loss', default='hybrid', choices=('l1', 'l2', 'hybrid'), help='training loss')
parser.add_argument('--alpha', type=float, default=0.1, help='hybrid loss coefficient')

# optimizer setting
parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
parser.add_argument('--scheduler_type', default='MultiStepLR', choices=('MultiStepLR', 'ReduceLROnPlateau'), help='learning rate decay type')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM'), help='optimizer to use (SGD | ADAM)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')

# model hyper-parameters
parser.add_argument('--model_title', default='EUNet', help='model name')
parser.add_argument('--n_iters', type=int, default=4, help='number of stages')
parser.add_argument('--n_modules', type=int, default=3, help='number of ESSG')
parser.add_argument('--n_feats', type=int, default=128, help='default number of filters')

parser.add_argument('--block_type', type=str, default=ESSB, help='block type')
parser.add_argument('--expand_ratio', type=int, default=2, help='expand ratio of ESSG')
parser.add_argument('--n_blocks', type=int, default=2, help='number of ESSB')
parser.add_argument('--dilations', type=int, default=[1, 2], help='dilation rates')

# test setting
parser.add_argument("--model_path", default="hsr/model/G.pth", type=str, help="path to checkpoint")

args = parser.parse_args()
