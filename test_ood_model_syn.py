"""
test model trained on synthetic dataset calibration
"""

import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
from utils.dc_utils import get_dataset, get_network, get_time, TensorDataset, epoch, ParamDiffAug, get_daparam
from utils.ood_utils import fpr_and_fdr_at_recall, get_ood_scores, get_measures
from utils.ood_utils import print_measures, print_measures_with_std, write_measures, write_measures_with_std
from utils.ood_utils import print_measures_pro, print_measures_with_std_pro, write_measures_pro, write_measures_with_std_pro


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--batch_test', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--data-path', type=str, default='/data4/sjma/dataset/CIFAR/', help='dataset path')
    parser.add_argument('--is_load_oe', action='store_true', default=False, help='load oe synthetic images')
    parser.add_argument('--load-path', type=str, default='result-model-synthetic', help='path to load results')
    parser.add_argument('--save-path', type=str, default='result-test-ood-synthetic', help='path to load results')
    parser.add_argument('--folder', type=str, required=True, help='folder to load synthetic dataset (e.g.: 20221108-150804)')
    parser.add_argument('--is_load_best', action='store_true', default=False, help='load best test acc checkpoints or last epoch')

    # number of loaded exps and models
    parser.add_argument('--epoch_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')

    parser.add_argument('--is-parallel', action='store_true', default=False, help='use torch data parallel')

    # OOD detection params
    parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
    parser.add_argument('--num_ood_ratio', type=int, default=1, help='number of ood ratio compared with id data.')
    parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
    parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
    parser.add_argument('--score', type=str, default='msp', help='OOD detection score function: [msp, mls, energy]')
    parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    args.num_classes = num_classes
    #trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_test, shuffle=False, num_workers=0)
    ood_num_examples = len(dst_test) // args.num_ood_ratio
    expected_ap = ood_num_examples / (ood_num_examples + len(dst_test))

    print('Hyper-parameters: \n', args.__dict__)
    if args.is_load_oe:
        args.load_path = 'result-model-synthetic-poe'
        args.save_path = 'result-test-ood-synthetic-poe'
    else:
        args.load_path = 'result-model-synthetic'
        args.save_path = 'result-test-ood-synthetic'
    # if args.use_xent:
    #     args.save_path = args.save_path + '-xent'
    if args.score != 'msp':
        print('OOD detection score function: ', args.score)
        args.save_path = args.save_path + '-' + args.score
    LOAD_PATH = os.path.join(args.load_path, args.method, args.dataset, args.model, str(args.ipc), args.folder)
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
    f_args.close()

    # load num exp net list
    net_list = []
    for exp in range(args.num_exp):
        Epoch = int(args.epoch_train)
        if args.is_load_best:
            load_path = os.path.join(LOAD_PATH, 'syn_trained_%s_%s_exp%d_best.pt' % (args.dataset, args.model, exp))
        else:
            load_path = os.path.join(LOAD_PATH, 'syn_trained_%s_%s_exp%d_epoch%d.pt' % (args.dataset, args.model, exp, Epoch))

        ckpts = torch.load(load_path)
        net = get_network(args.model, channel, num_classes, im_size).to(args.device)   # get a random model
        net = net.to(args.device)
        print('load from %s' % load_path)
        net.load_state_dict(ckpts)
        net.eval()
        net_list.append(net)



    # begin experiment
    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================

    print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')
    with open(log_save_path, 'w+') as f_log:
        f_log.write('\nUsing CIFAR-10 as typical data') if num_classes == 10 else f_log.write('\nUsing CIFAR-100 as typical data')
        f_log.write('\n')

    print('begin Accuracy and Error detection:')
    print(50 * '=')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('begin Accuracy and Error detection:')
        f_log.write('\n')
        f_log.write(50 * '=')
        f_log.write('\n')
    # ==================================================================================================================
    in_score_list = []
    error_rate_list = []
    err_auroc_list, err_aupr_list, err_fpr_list = [], [], []
    for i in range(args.num_exp):
        in_score, right_score, wrong_score = get_ood_scores(testloader, net_list[i], ood_num_examples, args, in_dist=True)
        in_score_list.append(in_score)
        num_right = len(right_score)
        num_wrong = len(wrong_score)
        # print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))
        # with open(log_save_path, 'a+') as f_log:
        #     f_log.write('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))
        #     f_log.write('\n')
        error_rate_list.append(100 * num_wrong / (num_wrong + num_right))

        '''error detection'''
        # print('\n\nError Detection')
        err_auroc, err_aupr, err_fpr = get_measures(wrong_score, right_score)
        err_auroc_list.append(err_auroc); err_aupr_list.append(err_aupr); err_fpr_list.append(err_fpr)
        # print_measures(err_auroc, err_aupr, err_fpr)
        # write_measures(err_auroc, err_aupr, err_fpr, file_path=log_save_path)

    print('Error Rate: {:.2f} +/- {:.2f}'.format(np.mean(error_rate_list), np.std(error_rate_list)))
    with open(log_save_path, 'a+') as f_log:
        f_log.write('Error Rate: {:.2f} +/- {:.2f}'.format(np.mean(error_rate_list), np.std(error_rate_list)))
        f_log.write('\n')
    print('\n\nError Detection')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nError Detection')
        f_log.write('\n')
    print_measures_with_std(err_auroc_list, err_aupr_list, err_fpr_list)
    write_measures_with_std(err_auroc_list, err_aupr_list, err_fpr_list, file_path=log_save_path)



    auroc_list, aupr_in_list, aupr_out_list, fpr_in_list, fpr_out_list = [], [], [], [], []
    auroc_v_list, aupr_in_v_list, aupr_out_v_list, fpr_in_v_list, fpr_out_v_list = [], [], [], [], []
    # utils function
    # ===============================================================================================
    def get_and_print_results(ood_loader):
        aurocs, auprs_in, auprs_out, fprs_in, fprs_out = [], [], [], [], []
        for i in range(args.num_exp):
            out_score = get_ood_scores(ood_loader, net_list[i], ood_num_examples, args)
            #measures = get_measures(out_score, in_score_list[i])
            # if args.out_as_pos:  # OE's defines out samples as positive
            #     measures = get_measures(out_score, in_score)
            # else:
            #     measures = get_measures(-in_score, -out_score)
            measures_in = get_measures(-in_score_list[i], -out_score)
            measures_out = get_measures(out_score, in_score_list[i])  # OE's defines out samples as positive
            aurocs.append(measures_in[0]); auprs_in.append(measures_in[1]); auprs_out.append(measures_out[1]); fprs_in.append(measures_in[2]); fprs_out.append(measures_out[2])

        auroc = np.mean(aurocs); aupr_in = np.mean(auprs_in); aupr_out = np.mean(auprs_out); fpr_in = np.mean(fprs_in); fpr_out = np.mean(fprs_out)
        auroc_list.append(auroc); aupr_in_list.append(aupr_in); aupr_out_list.append(aupr_out); fpr_in_list.append(fpr_in); fpr_out_list.append(fpr_out)

        if args.num_exp >= 3:
            print_measures_with_std_pro(aurocs, auprs_in, auprs_out, fprs_in, fprs_out)
            write_measures_with_std_pro(aurocs, auprs_in, auprs_out, fprs_in, fprs_out, file_path=log_save_path)
        else:
            print_measures_pro(auroc, aupr_in, aupr_out, fpr_in, fpr_out)
            write_measures_pro(auroc, aupr_in, aupr_out, fpr_in, fpr_out, file_path=log_save_path)


    def get_and_print_results_v(ood_loader):
        aurocs, auprs_in, auprs_out, fprs_in, fprs_out = [], [], [], [], []
        for i in range(args.num_exp):
            out_score = get_ood_scores(ood_loader, net_list[i], ood_num_examples, args)
            #measures = get_measures(out_score, in_score_list[i])
            # if args.out_as_pos:  # OE's defines out samples as positive
            #     measures = get_measures(out_score, in_score)
            # else:
            #     measures = get_measures(-in_score, -out_score)
            measures_in = get_measures(-in_score_list[i], -out_score)
            measures_out = get_measures(out_score, in_score_list[i])  # OE's defines out samples as positive
            aurocs.append(measures_in[0]); auprs_in.append(measures_in[1]); auprs_out.append(measures_out[1]); fprs_in.append(measures_in[2]); fprs_out.append(measures_out[2])

        auroc = np.mean(aurocs); aupr_in = np.mean(auprs_in); aupr_out = np.mean(auprs_out); fpr_in = np.mean(fprs_in); fpr_out = np.mean(fprs_out)
        auroc_v_list.append(auroc); aupr_in_v_list.append(aupr_in); aupr_out_v_list.append(aupr_out); fpr_in_v_list.append(fpr_in); fpr_out_v_list.append(fpr_out)

        if args.num_exp >= 3:
            print_measures_with_std_pro(aurocs, auprs_in, auprs_out, fprs_in, fprs_out)
            write_measures_with_std_pro(aurocs, auprs_in, auprs_out, fprs_in, fprs_out, file_path=log_save_path)
        else:
            print_measures_pro(auroc, aupr_in, aupr_out, fpr_in, fpr_out)
            write_measures_pro(auroc, aupr_in, aupr_out, fpr_in, fpr_out, file_path=log_save_path)
    # ===============================================================================================



    ''' ordinary OOD detection'''
    # ==================================================================================================================
    print('\n\nbegin ordinary OOD detection:')
    print(20 * '=')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nbegin ordinary OOD detection:')
        f_log.write('\n')
        f_log.write(20 * '=')
        f_log.write('\n')
    # # Gaussian Noise
    # dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
    # ood_data = torch.from_numpy(np.float32(np.clip(
    #     np.random.normal(size=(ood_num_examples * args.num_to_avg, 3, 32, 32), scale=0.5), -1, 1)))
    # ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    # ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_test, shuffle=True, num_workers=0)
    # print('\n\nGaussian Noise (sigma = 0.5) Detection')
    # with open(log_save_path, 'a+') as f_log:
    #     f_log.write('\n\nGaussian Noise (sigma = 0.5) Detection')
    #     f_log.write('\n')
    # get_and_print_results(ood_loader)

    # #  Uniform Noise
    # dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
    # ood_data = torch.from_numpy(
    #     np.random.uniform(size=(ood_num_examples * args.num_to_avg, 3, 32, 32),
    #                         low=-1.0, high=1.0).astype(np.float32))
    # ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    # ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_test, shuffle=True)
    # print('\n\nUniform[-1,1] Noise Detection')
    # with open(log_save_path, 'a+') as f_log:
    #     f_log.write('\n\nUniform[-1,1] Noise Detection')
    #     f_log.write('\n')
    # get_and_print_results(ood_loader)

    # # Rademacher Noise
    # dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
    # ood_data = torch.from_numpy(np.random.binomial(
    #     n=1, p=0.5, size=(ood_num_examples * args.num_to_avg, 3, 32, 32)).astype(np.float32)) * 2 - 1
    # ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    # ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_test, shuffle=True)
    # print('\n\nRademacher Noise Detection')
    # with open(log_save_path, 'a+') as f_log:
    #     f_log.write('\n\nRademacher Noise Detection')
    #     f_log.write('\n')
    # get_and_print_results(ood_loader)

    # # Blob
    # ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples * args.num_to_avg, 32, 32, 3)))
    # for i in range(ood_num_examples * args.num_to_avg):
    #     ood_data[i] = gblur(ood_data[i], sigma=1.5, channel_axis=False)
    #     ood_data[i][ood_data[i] < 0.75] = 0.0
    # dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
    # ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
    # ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    # ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_test, shuffle=True, num_workers=0)
    # print('\n\nBlob Detection')
    # with open(log_save_path, 'a+') as f_log:
    #     f_log.write('\n\nBlob Detection')
    #     f_log.write('\n')
    # get_and_print_results(ood_loader)

    # Textures
    ood_data = datasets.ImageFolder(root="/data4/sjma/dataset/OOD/dtd/images",
                                    transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32),
                                                                  transforms.ToTensor(), transforms.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_test, shuffle=True, num_workers=0)
    print('\n\nTexture Detection')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nTexture Detection')
        f_log.write('\n')
    get_and_print_results(ood_loader)

    # # SVHN
    # ood_data = datasets.SVHN('/data4/sjma/dataset/SVHN/', split='test', download=False,
    #                             transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean, std)]))
    # ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_test, shuffle=True, num_workers=0)
    # print('\n\nSVHN Detection')
    # with open(log_save_path, 'a+') as f_log:
    #     f_log.write('\n\nSVHN Detection')
    #     f_log.write('\n')
    # get_and_print_results(ood_loader)

    # Places
    ood_data = datasets.ImageFolder(root="/data4/sjma/dataset/OOD/places365",
                                    transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32),
                                    transforms.ToTensor(), transforms.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_test, shuffle=True, num_workers=0)
    print('\n\nPlaces Detection')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nPlaces Detection')
        f_log.write('\n')
    get_and_print_results(ood_loader)

    # # TinyImageNet-C
    # ood_data = datasets.ImageFolder(root="/data4/sjma/dataset/OOD/Imagenet",
    #                                 transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32),
    #                                                               transforms.ToTensor(), transforms.Normalize(mean, std)]))
    # ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_test, shuffle=True, num_workers=0)
    # print('\n\nTinyImageNet-crop Detection')
    # with open(log_save_path, 'a+') as f_log:
    #     f_log.write('\n\nTinyImageNet-crop Detection')
    #     f_log.write('\n')
    # get_and_print_results(ood_loader)

    # TinyImageNet-R
    ood_data = datasets.ImageFolder(root="/data4/sjma/dataset/OOD/Imagenet_resize",
                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_test, shuffle=True, num_workers=0)
    print('\n\nTinyImageNet-resize Detection')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nTinyImageNet-resize Detection')
        f_log.write('\n')
    get_and_print_results(ood_loader)

    # # LSUN-C
    # ood_data = datasets.ImageFolder(root="/data4/sjma/dataset/OOD/LSUN",
    #                                 transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32),
    #                                                               transforms.ToTensor(), transforms.Normalize(mean, std)]))
    # ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_test, shuffle=True, num_workers=0)
    # print('\n\nLSUN-crop Detection')
    # with open(log_save_path, 'a+') as f_log:
    #     f_log.write('\n\nLSUN-crop Detection')
    #     f_log.write('\n')
    # get_and_print_results(ood_loader)

    # LSUN-R
    ood_data = datasets.ImageFolder(root="/data4/sjma/dataset/OOD/LSUN_resize",
                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_test, shuffle=True, num_workers=0)
    print('\n\nLSUN-resize Detection')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nLSUN-resize Detection')
        f_log.write('\n')
    get_and_print_results(ood_loader)

    # iSUN
    ood_data = datasets.ImageFolder(root="/data4/sjma/dataset/OOD/iSUN",
                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_test, shuffle=True, num_workers=0)
    print('\n\niSUN Detection')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\niSUN Detection')
        f_log.write('\n')
    get_and_print_results(ood_loader)

    # CIFAR data
    # mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    # std = [x / 255 for x in [63.0, 62.1, 66.7]]
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if args.dataset == 'CIFAR10':
        ood_data = datasets.CIFAR100('/data4/sjma/dataset/CIFAR/', train=False, transform=test_transform)
    else:
        ood_data = datasets.CIFAR10('/data4/sjma/dataset/CIFAR/', train=False, transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_test, shuffle=True, num_workers=0)
    print('\n\nCIFAR-100 Detection') if args.dataset == 'CIFAR10' else print('\n\nCIFAR-10 Detection')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nCIFAR-100 Detection') if args.dataset == 'CIFAR10' else f_log.write('\n\nCIFAR-10 Detection')
        f_log.write('\n')
    get_and_print_results(ood_loader)



    '''validation ood'''
    # ==================================================================================================================
    if not args.validate:
        # print mean results over exps
        # ==================================================================================================================
        print('\n\n\nMean results:')
        print('\n\nMean Error Detection Results')
        with open(log_save_path, 'a+') as f_log:
            f_log.write('\n\n\nMean results:')
            f_log.write('\n')
            f_log.write('\n\nMean Error Detection Results')
            f_log.write('\n')
        print_measures_with_std(err_auroc_list, err_aupr_list, err_fpr_list)
        write_measures_with_std(err_auroc_list, err_aupr_list, err_fpr_list, file_path=log_save_path)

        print('\n\nMean (Ordinary) OOD Detection Results')
        with open(log_save_path, 'a+') as f_log:
            f_log.write('\n\nMean (Ordinary) OOD Detection Results')
            f_log.write('\n')
        # print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)
        print_measures_with_std_pro(auroc_list, aupr_in_list, aupr_out_list, fpr_in_list, fpr_out_list)
        write_measures_with_std_pro(auroc_list, aupr_in_list, aupr_out_list, fpr_in_list, fpr_out_list, file_path=log_save_path)

        exit()

    print('begin validation OOD detection:')
    print(50 * '=')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('begin validation OOD detection:')
        f_log.write('\n')
        f_log.write(50 * '=')
        f_log.write('\n')


    #  Arithmetic Mean of Images
    if args.dataset == 'CIFAR10':
        ood_data = datasets.CIFAR100('/data4/sjma/dataset/CIFAR/', train=False, transform=test_transform)
    else:
        ood_data = datasets.CIFAR10('/data4/sjma/dataset/CIFAR/', train=False, transform=test_transform)


    class AvgOfPair(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.shuffle_indices = np.arange(len(dataset))
            np.random.shuffle(self.shuffle_indices)

        def __getitem__(self, i):
            random_idx = np.random.choice(len(self.dataset))
            while random_idx == i:
                random_idx = np.random.choice(len(self.dataset))

            return self.dataset[i][0] / 2. + self.dataset[random_idx][0] / 2., 0

        def __len__(self):
            return len(self.dataset)


    ood_loader = torch.utils.data.DataLoader(AvgOfPair(ood_data),
                                             batch_size=args.batch_test, shuffle=True, num_workers=0)
    print('\n\nArithmetic Mean of Random Image Pair Detection')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nArithmetic Mean of Random Image Pair Detection')
        f_log.write('\n')
    get_and_print_results_v(ood_loader)

    # Geometric Mean of Images
    if args.dataset == 'CIFAR10':
        ood_data = datasets.CIFAR100('/data4/sjma/dataset/CIFAR/', train=False, transform=transforms.ToTensor())
    else:
        ood_data = datasets.CIFAR10('/data4/sjma/dataset/CIFAR/', train=False, transform=transforms.ToTensor())


    class GeomMeanOfPair(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.shuffle_indices = np.arange(len(dataset))
            np.random.shuffle(self.shuffle_indices)

        def __getitem__(self, i):
            random_idx = np.random.choice(len(self.dataset))
            while random_idx == i:
                random_idx = np.random.choice(len(self.dataset))

            return transforms.Normalize(mean, std)(torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0])), 0

        def __len__(self):
            return len(self.dataset)

    ood_loader = torch.utils.data.DataLoader(
        GeomMeanOfPair(ood_data), batch_size=args.batch_test, shuffle=True, num_workers=0)
    print('\n\nGeometric Mean of Random Image Pair Detection')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nGeometric Mean of Random Image Pair Detection')
        f_log.write('\n')
    get_and_print_results_v(ood_loader)

    # Jigsaw Image
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_test, shuffle=True, num_workers=0)
    jigsaw = lambda x: torch.cat((
        torch.cat((torch.cat((x[:, 8:16, :16], x[:, :8, :16]), 1),
                    x[:, 16:, :16]), 2),
        torch.cat((x[:, 16:, 16:],
                    torch.cat((x[:, :16, 24:], x[:, :16, 16:24]), 2)), 2),
    ), 1)
    ood_loader.dataset.transform = transforms.Compose([transforms.ToTensor(), jigsaw, transforms.Normalize(mean, std)])
    print('\n\nJigsawed Images Detection')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nJigsawed Images Detection')
        f_log.write('\n')
    get_and_print_results_v(ood_loader)

    # Speckled Images
    speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
    ood_loader.dataset.transform = transforms.Compose([transforms.ToTensor(), speckle, transforms.Normalize(mean, std)])
    print('\n\nSpeckle Noised Images Detection')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nSpeckle Noised Images Detection')
        f_log.write('\n')
    get_and_print_results_v(ood_loader)

    # Pixelated Images
    #pixelate = lambda x: x.resize((int(32 * 0.2), int(32 * 0.2)), PILImage.BOX).resize((32, 32), PILImage.BOX)
    pixelate = lambda x: x.resize((int(32 * 0.2), int(32 * 0.2)), PILImage.Resampling.BOX).resize((32, 32), PILImage.Resampling.BOX)
    ood_loader.dataset.transform = transforms.Compose([pixelate, transforms.ToTensor(), transforms.Normalize(mean, std)])
    print('\n\nPixelate Detection')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nPixelate Detection')
        f_log.write('\n')
    get_and_print_results_v(ood_loader)

    # RGB Ghosted/Shifted Images
    rgb_shift = lambda x: torch.cat((x[1:2].index_select(2, torch.LongTensor([i for i in range(32 - 1, -1, -1)])),
                                    x[2:, :, :], x[0:1, :, :]), 0)
    ood_loader.dataset.transform = transforms.Compose([transforms.ToTensor(), rgb_shift, transforms.Normalize(mean, std)])
    print('\n\nRGB Ghosted/Shifted Image Detection')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nRGB Ghosted/Shifted Images Detection')
        f_log.write('\n')
    get_and_print_results_v(ood_loader)

    # Inverted Images
    # not done on all channels to make image ood with higher probability
    #invert = lambda x: torch.cat((x[0:1, :, :], 1 - x[1:2, :, ], 1 - x[2:, :, :],), 0)
    invert = lambda x: torch.cat((x[0:1, :, :], 1 - x[1:2, :, :], 1 - x[2:, :, :],), 0)
    ood_loader.dataset.transform = transforms.Compose([transforms.ToTensor(), invert, transforms.Normalize(mean, std)])
    print('\n\nInverted Image Detection')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nInverted Images Detection')
        f_log.write('\n')
    get_and_print_results_v(ood_loader)



    # print mean results over exps
    # ==================================================================================================================
    print('\n\n\nMean results:')
    print('\n\nMean Error Detection Results')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\n\nMean results:')
        f_log.write('\n')
        f_log.write('\n\nMean Error Detection Results')
        f_log.write('\n')
    print_measures_with_std(err_auroc_list, err_aupr_list, err_fpr_list)
    write_measures_with_std(err_auroc_list, err_aupr_list, err_fpr_list, file_path=log_save_path)

    print('\n\nMean (Ordinary) OOD Detection Results')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nMean (Ordinary) OOD Detection Results')
        f_log.write('\n')
    # print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)
    print_measures_with_std_pro(auroc_list, aupr_in_list, aupr_out_list, fpr_in_list, fpr_out_list)
    write_measures_with_std_pro(auroc_list, aupr_in_list, aupr_out_list, fpr_in_list, fpr_out_list, file_path=log_save_path)

    print('\n\nMean (Validation) OOD Detection Results')
    with open(log_save_path, 'a+') as f_log:
        f_log.write('\n\nMean (Validation) OOD Detection Results')
        f_log.write('\n')
    # print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)
    print_measures_with_std_pro(auroc_v_list, aupr_in_v_list, aupr_out_v_list, fpr_in_v_list, fpr_out_v_list)
    write_measures_with_std_pro(auroc_v_list, aupr_in_v_list, aupr_out_v_list, fpr_in_v_list, fpr_out_v_list, file_path=log_save_path)


if __name__ == '__main__':
    main()
