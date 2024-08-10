import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils.dc_utils import get_loops, get_dataset, get_network, get_eval_pool, get_daparam, get_time, TensorDataset, DiffAugment, ParamDiffAug, augment
from utils.dc_utils import epoch_oe


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')   # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='/data4/sjma/dataset/CIFAR/', help='dataset path')
    parser.add_argument('--load_path', type=str, default='result', help='path to save results')
    parser.add_argument('--is_load_oe', action='store_true', default=False, help='load oe synthetic images')
    parser.add_argument('--save_path', type=str, default='result-model-synthetic', help='path to load results')
    parser.add_argument('--folder', type=str, required=True, help='folder to load synthetic dataset (e.g.: 20221108-150804)')
    parser.add_argument('--exp_idx', type=int, default=0, help='experiment idx to load synthetic dataset')
    parser.add_argument('--lambda_oe', type=float, default=0.5, help='oe loss weight')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if (args.method == 'DSA' or args.method == 'DM') else False


    channel, im_size, num_classes, class_names, mean, std, dst_train_real, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    '''save results setting'''
    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)
    if args.is_load_oe:
        args.load_path = 'result-poe'
        args.save_path = 'result-model-synthetic-poe'
    else:
        args.load_path = 'result'
        args.save_path = 'result-model-synthetic'
    load_dir = os.path.join(args.load_path, args.method, args.dataset, args.model, str(args.ipc), args.folder)
    file_name = 'res_%s_%s_%s_%dipc.pt' % (args.method, args.dataset, args.model, args.ipc)
    LOAD_PATH = os.path.join(load_dir, file_name)
    SAVE_PATH = os.path.join(args.save_path, args.method, args.dataset, args.model, str(args.ipc), args.folder)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    log_save_path = os.path.join(SAVE_PATH, 'logs.txt')
    args_save_path = os.path.join(SAVE_PATH, 'args.txt')
    f_args = open(args_save_path, 'w')
    f_args.write('args: \n')
    f_args.write(str(vars(args)))
    f_args.write('\n')
    f_args.write('Hyper-parameters: \n')
    f_args.write(str(args.__dict__))
    f_args.write('\n\n\n')
    f_args.write('Evaluation model pool: ')
    f_args.write(str(model_eval_pool))
    f_args.close()


    # load checkpoints
    print('load synthetic dataset from: %s' % LOAD_PATH)
    ckpts = torch.load(LOAD_PATH)
    data = ckpts['data']
    data_oe = ckpts['data_oe']
    num_exp = len(data)
    print('total %d experiments in the checkpoints, load from exp idx %d' % (num_exp, args.exp_idx))

    load_image_syn, load_label_syn = data[args.exp_idx]
    image_syn, label_syn = load_image_syn.to(args.device), load_label_syn.to(args.device)
    load_image_syn_oe, load_label_syn_oe = data_oe[args.exp_idx]
    image_syn_oe, label_syn_oe = load_image_syn_oe.to(args.device), load_label_syn_oe.to(args.device)
    print(image_syn.size())
    print(image_syn_oe.size())
    # synthetic dst_train
    dst_train = TensorDataset(image_syn, label_syn)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
    dst_train_oe = TensorDataset(image_syn_oe, label_syn_oe)
    trainloader_oe = torch.utils.data.DataLoader(dst_train_oe, batch_size=args.batch_train, shuffle=True, num_workers=0)


    '''1. loop for multiple model_eval'''
    for model_eval in model_eval_pool:
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s' % (args.model, model_eval))
        with open(log_save_path, 'a+') as f_log:
            f_log.write('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s' % (args.model, model_eval))
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


        '''2. loop for multiple exps on each model_eval'''
        for it_eval in range(args.num_eval):
            print('eval iter: %d' % (it_eval + 1))
            with open(log_save_path, 'a+') as f_log:
                f_log.write('eval iter: %d' % (it_eval + 1))
                f_log.write('\n')
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)   # get a random model

            lr = float(args.lr_net)
            Epoch = int(args.epoch_eval_train)
            lr_schedule = [Epoch // 2 + 1]
            optimizer = torch.optim.SGD(net_eval.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            criterion = nn.CrossEntropyLoss().to(args.device)

            start = time.time()
            '''3. loop for epoch'''
            best_acc = 0.0
            for ep in range(Epoch + 1):
                loss_train, acc_train = epoch_oe('train', trainloader, trainloader_oe, net_eval, optimizer, criterion, args, aug=True)

                # test model in specific epochs
                if ep % 50 == 0 or ep == Epoch:
                    loss_test, acc_test = epoch_oe('test', testloader, None, net_eval, optimizer, criterion, args, aug=False)

                    if acc_test > best_acc:
                        best_acc = acc_test
                        print('congrats! best_test_acc! epoch = %04d test acc = %.4f' % (ep, acc_test))
                        with open(log_save_path, 'a+') as f_log:
                            f_log.write('congrats! best_test_acc! epoch = %04d test acc = %.4f' % (ep, acc_test))
                            f_log.write('\n')
                        if ep >= 100:
                            torch.save(net_eval.state_dict(), os.path.join(SAVE_PATH, 'syn_trained_%s_%s_exp%d_best.pt' % (args.dataset, model_eval, it_eval)))

                    if ep % 200 == 0 or ep == Epoch:
                        time_train = time.time() - start
                        print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, ep, int(time_train), loss_train, acc_train, acc_test))
                        with open(log_save_path, 'a+') as f_log:
                            f_log.write('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, ep, int(time_train), loss_train, acc_train, acc_test))
                            f_log.write('\n')
                        # save ckpts
                        if ep >= (3 / 5 * Epoch):
                            torch.save(net_eval.state_dict(), os.path.join(SAVE_PATH, 'syn_trained_%s_%s_exp%d_epoch%d.pt' % (args.dataset, model_eval, it_eval, ep)))

                if ep in lr_schedule:
                    lr *= 0.1
                    optimizer = torch.optim.SGD(net_eval.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

            # loss_test, acc_test = epoch('test', testloader, net_eval, optimizer, criterion, args, aug=False)
            # print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))
            # with open(log_save_path, 'a+') as f_log:
            #     f_log.write('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))
            #     f_log.write('\n')

            #torch.save(net_eval.state_dict(), os.path.join(SAVE_PATH, 'syn_trained_%s_%s_exp%d.pt' % (args.dataset, model_eval, it_eval)))
            #accs.append(acc_test)
            accs.append(best_acc)

        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs), model_eval, np.mean(accs), np.std(accs)))
        with open(log_save_path, 'a+') as f_log:
            f_log.write('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs), model_eval, np.mean(accs), np.std(accs)))
            f_log.write('\n')



if __name__ == '__main__':
    main()
