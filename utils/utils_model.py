import os
import re
import glob
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel


def find_last_checkpoint(save_dir, net_type='G'):
    """
    Args:
        save_dir: model folder
        net_type: 'G' or 'D' or 'optimizerG' or 'optimizerD'

    Return:
        init_iter: iteration number
        init_path: model path
    """
    file_list = glob.glob(os.path.join(save_dir, '*_{}.pth'.format(net_type)))
    if file_list:
        iter_exist = []
        for file_ in file_list:
            iter_current = re.findall(r"(\d+)_{}.pth".format(net_type), file_)
            iter_exist.append(int(iter_current[0]))
        init_iter = max(iter_exist)
        init_path = os.path.join(save_dir, '{}_{}.pth'.format(init_iter, net_type))
    else:
        init_iter = 0
        init_path = None
    return init_iter, init_path


# ----------------------------------------
# Get bare model, especially under wrapping with
# DistributedDataParallel or DataParallel.
# ----------------------------------------
def get_bare_model(network):
    if isinstance(network, (DataParallel, DistributedDataParallel)):
        network = network.module
    return network


# ----------------------------------------
# save the state_dict of the network
# ----------------------------------------
def save_network(save_dir, network, network_label, iter_label):
    save_filename = '{}_{}.pth'.format(iter_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    network = get_bare_model(network)
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)


# ----------------------------------------
# load the state_dict of the network
# ----------------------------------------
def load_network(load_path, network):
    network = get_bare_model(network)
    state_dict = torch.load(load_path)
    network.load_state_dict(state_dict)
