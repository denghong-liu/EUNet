import os
import math
import random
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from option import args
from data import HSRData
from models import EUNet
from models.model_plain import ModelPlain
from utils import utils_model
from utils import utils_image as util


warnings.filterwarnings("ignore")


def main():
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    args.ckp_dir = os.path.join(args.save_dir, 'model')
    util.mkdir(args.ckp_dir)
    args.log_dir = os.path.join(args.save_dir, 'log')
    util.mkdir(args.log_dir)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    util.write_python_file('option.py', os.path.join(args.log_dir, 'config.txt'))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    if not args.seed:
        args.seed = random.randint(1, 10000)
    print("Random seed: ", args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    if args.resume:
        args.init_path_G = args.resume
        args.init_path_optimizerG = None
        current_step = 0
    else:
        init_iter_G, args.init_path_G = utils_model.find_last_checkpoint(args.ckp_dir, net_type='G')
        init_iter_optimizerG, args.init_path_optimizerG = utils_model.find_last_checkpoint(args.ckp_dir, net_type='optimizerG')
        current_step = max(init_iter_G, init_iter_optimizerG)


    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and val
    # 3) create running parameters
    # ----------------------------------------
    print('===> Loading datasets')
    train_path = args.dir_data + args.dataset_name + '/train/X' + str(args.scale) + '/patches'
    train_set = HSRData(data_dir=train_path, sigma=args.sigma, augment=True)
    print('Dataset [{:s} - train_dataset] is created.'.format(train_set.__class__.__name__))
    train_size = int(math.ceil(len(train_set) / args.batch_size))
    print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.n_threads,
                              drop_last=False,
                              pin_memory=False)

    val_path = args.dir_data + args.dataset_name + '/val/X' + str(args.scale) + '/patches'
    val_set = HSRData(data_dir=val_path, sigma=args.sigma, augment=False)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.n_threads,
                            drop_last=False,
                            pin_memory=False)

    args.milestones = [100 * train_size]  # learning rate decay with iterations
    args.print_every = 1 * train_size  # how many batches to wait before logging training status
    args.test_every = 5 * train_size  # do test per every N epochs
    args.save_every = 5 * train_size  # save intermediate models per every N epochs


    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    print('===> Building model')
    network = EUNet(scale=args.scale, n_iter=args.n_iters, n_colors=args.n_colors, n_feats=args.n_feats,
                  n_modules=args.n_modules, block=args.block_type, n_blocks=args.n_blocks, dilations=args.dilations,
                  expand_ratio=args.expand_ratio, is_blur=args.is_blur)
    print('Training model [{:s}] is created.'.format(network.__class__.__name__))
    print(network)
    model = ModelPlain(opt=args, netG=network)
    model.init_train()


    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    print('===> Start training')
    for epoch in range(args.epochs):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % args.print_every == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                          model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                print(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % args.save_every == 0:
                print('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % args.test_every == 0:

                val_loss = 0.0
                idx = 0

                for val_data in val_loader:
                    idx += 1
                    model.feed_data(val_data)
                    G_loss = model.test()
                    val_loss += G_loss

                val_loss = val_loss / idx
                # testing log
                print('<epoch:{:3d}, iter:{:8,d}, Average loss : {:<.3e}\n'.format(epoch, current_step, val_loss))


if __name__ == "__main__":
    main()
