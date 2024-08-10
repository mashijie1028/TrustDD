"""
dataset condensation using Outlier Exposure

date: 2023-02-06

"""

import os
import time
import copy
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils.dc_utils import get_loops, get_dataset, get_network, get_eval_pool, get_daparam, match_loss, get_time, TensorDataset, DiffAugment, ParamDiffAug, augment
from utils.dc_utils import epoch_oe, evaluate_synset_oe
from utils.randomimages_300k_loader import RandomImages


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--data_path', type=str, default='/data4/sjma/dataset/CIFAR/', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result-oe', help='path to save results')

    # exp args
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')   # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')

    # opt args
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--is-parallel', action='store_true', default=False, help='use torch data parallel')

    # dc hyperparams args
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')

    # OE args
    parser.add_argument('--oe_data_path', type=str, default='/data4/sjma/dataset/OOD/300K_random_images.npy', help='outlier dataset path')
    parser.add_argument('--num_oe_per_class', type=int, default=10000, help='number of oe samples corresponding to each class')
    parser.add_argument('--batch_oe', type=int, default=256, help='batch size for outlier')
    parser.add_argument('--lambda_oe', type=float, default=0.5, help='oe loss weight')
    parser.add_argument('--seed', type=int, default=1, help='random seed for numpy')

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    eval_it_pool = np.arange(0, args.Iteration + 1, 500).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration]   # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    # oe data
    oe_dataset = RandomImages(transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]))
    oe_loader = torch.utils.data.DataLoader(oe_dataset, batch_size=args.batch_oe, shuffle=False, num_workers=0)
    print(len(oe_dataset))

    accs_all_exps = dict()   # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    data_save_oe = []

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    '''save results setting'''
    TIME_NOW = str(time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    print(TIME_NOW)
    SAVE_PATH = os.path.join(args.save_path, args.method, args.dataset, args.model, str(args.ipc), TIME_NOW)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    log_save_path = os.path.join(SAVE_PATH, 'logs.txt')
    # save args.txt
    args_save_path = os.path.join(SAVE_PATH, 'args.txt')
    f_args = open(args_save_path, 'w')
    f_args.write('args: \n')
    f_args.write(str(vars(args)))
    f_args.write('\n')
    f_args.write('Hyper-parameters: \n')
    f_args.write(str(args.__dict__))
    f_args.write('\n\n\n')
    f_args.write('Evaluation iteration pool: ')
    f_args.write(str(eval_it_pool))
    f_args.write('\n\n\n')
    f_args.write('Evaluation model pool: ')
    f_args.write(str(model_eval_pool))
    f_args.close()

    for exp in range(args.num_exp):
        print('================== Exp %d ==================' % exp)
        with open(log_save_path, 'a+') as f_log:
            f_log.write('================== Exp %d ==================' % exp)
            f_log.write('\n')

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            print('class c = %d: %d real images' % (c, len(indices_class[c])))

        def get_images(c, n):   # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f' % (ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        '''organize the outlier exposure (oe) dataset'''
        images_oe_all = []
        images_oe_all = [torch.unsqueeze(oe_dataset[i][0], dim=0) for i in range(len(oe_dataset))]
        images_oe_all = torch.cat(images_oe_all, dim=0).to(args.device)

        idx_oe_shuffle = [[] for _ in range(num_classes)]
        idx_oe_shuffle_all = list(range(len(oe_dataset)))
        random.shuffle(idx_oe_shuffle_all)
        for i in range(num_classes):
            #idx_oe_shuffle[i] = np.random.permutation(args.num_oe_per_class)
            #idx_oe_shuffle[i] = np.random.permutation(len(oe_dataset))
            idx_oe_shuffle[i] = idx_oe_shuffle_all[i*args.num_oe_per_class:(i+1)*args.num_oe_per_class]   # NOTE!

        #print('index oe shuffle:', idx_oe_shuffle)
        # with open(log_save_path, 'a+') as f_log:
        #     f_log.write('index oe shuffle:')
        #     f_log.write('\n')
        #     f_log.write(str(idx_oe_shuffle))
        #     f_log.write('\n')

        def get_images_oe(c, n):   # get random n images from outlier dataset
            #idx_shuffle = np.random.permutation(len(oe_dataset))[:n]
            #idx_shuffle_ = idx_oe_shuffle[c][:n]
            idx_shuffle_ = np.random.permutation(idx_oe_shuffle[c])[:n]   # NOTE!
            return images_oe_all[idx_shuffle_]


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc, dtype=np.int64) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)   # [0,0,0, 1,1,1, ..., 9,9,9]

        '''initialize the synthetic oe data'''
        image_syn_oe = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn_oe = torch.tensor([np.zeros(args.ipc, dtype=np.int64) for _ in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)   # dummy labels

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
                image_syn_oe.data[c * args.ipc:(c + 1) * args.ipc] = get_images_oe(c, args.ipc).detach().data

        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)   # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        optimizer_img_oe = torch.optim.SGD([image_syn_oe, ], lr=args.lr_img, momentum=0.5)   # optimizer_img for synthetic oe data
        optimizer_img_oe.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins' % get_time())
        with open(log_save_path, 'a+') as f_log:
            f_log.write('%s training begins' % get_time())
            f_log.write('\n')

        for it in range(args.Iteration + 1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, it))
                    with open(log_save_path, 'a+') as f_log:
                        f_log.write('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, it))
                        f_log.write('\n')
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                        with open(log_save_path, 'a+') as f_log:
                            f_log.write('DSA augmentation strategy: \n')
                            f_log.write(args.dsa_strategy)
                            f_log.write('\n')
                            f_log.write('DSA augmentation parameters: \n')
                            f_log.write(str(args.dsa_param.__dict__))
                            f_log.write('\n')
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)   # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        print('DC augmentation parameters: \n', args.dc_aug_param)
                        with open(log_save_path, 'a+') as f_log:
                            f_log.write('DC augmentation parameters: \n')
                            f_log.write(str(args.dc_aug_param))
                            f_log.write('\n')

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)   # get a random model
                        if args.is_parallel:
                            net_eval = nn.DataParallel(net_eval)
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())   # avoid any unaware modification
                        image_syn_oe_eval, label_syn_oe_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())   # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset_oe(it_eval, net_eval, image_syn_eval, label_syn_eval, image_syn_oe_eval, label_syn_oe_eval, testloader, args)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs), model_eval, np.mean(accs), np.std(accs)))
                    with open(log_save_path, 'a+') as f_log:
                        f_log.write('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs), model_eval, np.mean(accs), np.std(accs)))
                        f_log.write('\n')

                    if it == args.Iteration:   # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(SAVE_PATH, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png' % (args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis < 0] = 0.0
                image_syn_vis[image_syn_vis > 1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc)   # Trying normalize = True/False may get better visual effects.

                save_name = os.path.join(SAVE_PATH, 'vis_oe_%s_%s_%s_%dipc_exp%d_iter%d.png' % (args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn_oe.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis < 0] = 0.0
                image_syn_vis[image_syn_vis > 1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc)   # Trying normalize = True/False may get better visual effects.


            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)   # get a random model
            if args.is_parallel:
                net = nn.DataParallel(net)
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg_cls = 0
            loss_avg_oe = 0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.


            for ol in range(args.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():   #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train()   # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real)   # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval()   # fix mu and sigma of every BatchNorm layer


                ''' update synthetic data '''
                # 1. real data
                # ======================================================================================================
                loss_cls = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c * args.ipc: (c + 1) * args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)

                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss_cls += match_loss(gw_syn, gw_real, args)

                optimizer_img.zero_grad()
                loss_cls.backward()
                optimizer_img.step()
                loss_avg_cls += loss_cls.item()

                # 2. oe data
                # ======================================================================================================
                loss_oe = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    # oe data
                    img_real_oe = get_images_oe(c, args.batch_oe)
                    img_syn_oe = image_syn_oe[c * args.ipc: (c + 1) * args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real_oe = DiffAugment(img_real_oe, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        # aug oe
                        img_syn_oe = DiffAugment(img_syn_oe, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real_oe = net(img_real_oe)
                    # oe loss
                    loss_real_oe = - (output_real_oe.mean(1) - torch.logsumexp(output_real_oe, dim=1)).mean()

                    gw_real = torch.autograd.grad(loss_real_oe, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn_oe = net(img_syn_oe)
                    loss_syn_oe = - (output_syn_oe.mean(1) - torch.logsumexp(output_syn_oe, dim=1)).mean()
                    gw_syn = torch.autograd.grad(loss_syn_oe, net_parameters, create_graph=True)

                    loss_oe += match_loss(gw_syn, gw_real, args)   # NOTE!

                optimizer_img_oe.zero_grad()
                loss_oe.backward()
                optimizer_img_oe.step()
                loss_avg_oe += loss_oe.item()

                if ol == args.outer_loop - 1:
                    break


                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                image_syn_train_oe, label_syn_train_oe = copy.deepcopy(image_syn_oe.detach()), copy.deepcopy(label_syn_oe.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                dst_syn_train_oe = TensorDataset(image_syn_train_oe, label_syn_train_oe)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                trainloader_oe = torch.utils.data.DataLoader(dst_syn_train_oe, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch_oe('train', trainloader, trainloader_oe, net, optimizer_net, criterion, args, aug=True if args.dsa else False)


            loss_avg_cls /= (num_classes * args.outer_loop)
            loss_avg_oe /= (num_classes * args.outer_loop)

            if it % 10 == 0:
                print('%s iter = %04d, loss_cls = %.4f, loss_oe = %.4f' % (get_time(), it, loss_avg_cls, loss_avg_oe))
                with open(log_save_path, 'a+') as f_log:
                    f_log.write('%s iter = %04d, loss_cls = %.4f, loss_oe = %.4f' % (get_time(), it, loss_avg_cls, loss_avg_oe))
                    f_log.write('\n')

            if it == args.Iteration:   # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                data_save_oe.append([copy.deepcopy(image_syn_oe.detach().cpu()), copy.deepcopy(label_syn_oe.detach().cpu())])
                torch.save({'data': data_save, 'data_oe': data_save_oe, 'accs_all_exps': accs_all_exps, }, os.path.join(SAVE_PATH, 'res_%s_%s_%s_%dipc.pt' % (args.method, args.dataset, args.model, args.ipc)))


    print('\n==================== Final Results ====================')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n==================== Final Results ====================')
        f_log.write('\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' % (args.num_exp, args.model, len(accs), key, np.mean(accs) * 100, np.std(accs) * 100))
        with open(log_save_path, 'a+') as f_log:
            f_log.write('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' % (args.num_exp, args.model, len(accs), key, np.mean(accs) * 100, np.std(accs) * 100))
            f_log.write('\n')



if __name__ == '__main__':
    main()
