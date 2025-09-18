import argparse
import logging
import math
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from utils import DiceLoss, MyLoss_correction, get_cov_mask, S2_Loss, \
    DiceLoss_amos, ce_amos, MyLoss_correction_amos, \
    DiceLoss_flare, ce_flare, MyLoss_correction_flare
from torchvision import transforms
from utils import test_single_volume
from reweighting import weight_learner
from STG import STG
from scipy.ndimage import zoom
from sklearn import metrics
from torch.nn import functional as F

# class_weight_camvid=torch.tensor([0.25652730831033116,0.18528266685745448,4.396287575365375,0.1368693220338383,
#                                   0.9184731310542199,0.38731986379829597,3.5330742906141994,0.8126852672146507,
#                                   1.7246197983929721]).cuda()
def pred_thresh(net, num_classes, out, label_batch, batch):
    patch_size = [224, 224]

    for i in range(batch):
        image = out[i, :, :]
        # image = torch.argmax(torch.softmax(image, dim=0), dim=0).float()
        label = label_batch[i, :, :]
        # image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
        # if len(image.shape) == 3:
        #     # evidences = np.zeros_like(label)
        #     for ind in range(image.shape[0]):
        #         slice = image[ind, :, :]
        #         slice_label = label[ind, :, :]
        #         x, y = slice.shape[0], slice.shape[1]
        #         if x != patch_size[0] or y != patch_size[1]:
        #             slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
        #             slice_label = zoom(slice_label, (patch_size[0] / x, patch_size[1] / y), order=3)
        #         input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        # net.eval()
        # max = slice_label.max()
        # t = np.sum(slice_label)
        # if slice_label.max() != 0:
        # with torch.no_grad():
        # outputs = net(image)
        e = [F.softplus(image)]
        alpha = dict()
        alpha[0] = e[0] + 1

        S = torch.sum(alpha[0], dim=0, keepdim=True)
        E = alpha[0] - 1
        b = E / (S.expand(E.shape))

        u = num_classes / S

        pred = torch.argmax(torch.softmax(image, dim=0), dim=0).squeeze(0)
        # pred = image.cpu().detach().numpy()

        u = u.view(-1).tolist()
        un_gt = 1 - torch.eq(torch.tensor(pred), torch.tensor(label)).float()
        un_gt = un_gt.view(-1).tolist()


                    # u_list = list(np.array(u_list).flatten())
                    # u_label_list = list(np.array(u_label_list).flatten().astype('int'))
                    # fpr_Pri, tpr_Pri, thresh = metrics.roc_curve(u_label_list, u_list)
                    # tpr_Pri = np.nan_to_num(tpr_Pri)

                    # tpr = TPR(u_label_list, u_list)

                    # print(thresh)

                # out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
                # out = out.cpu().detach().numpy()
                # if x != patch_size[0] or y != patch_size[1]:
                #     pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                # else:
                #     pred = out
                # evidences[ind] = pred

        # for idx in range(image.shape[0]):

    # u_list = list(np.array(u_list).flatten())
    # u_label_list = list(np.array(u_label_list).flatten().astype('int'))
    #
    # fpr_Pri, tpr_Pri, thresh = metrics.roc_curve(u_label_list, u_list)
    # # tprs = TPR(u_label_list, u_list)
    # tpr_Pri = np.nan_to_num(tpr_Pri)
    # max_j = max(zip(fpr_Pri, tpr_Pri), key=lambda x: 2 * x[1] - x[0])
    # # max_th[i_batch] = max_j
    # pred_thresh = thresh[list(zip(fpr_Pri, tpr_Pri)).index(max_j)]
    # print("opt_pred ===== {}".format(pred_thresh))
    # # print(np.mean(max_th))

    return u, un_gt

def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)   #gao

def _one_hot_encoder(input_tensor, n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def s2_change(target, n_classes):
    target = _one_hot_encoder(target, n_classes)
    B = target.size(0)
    target_s2 = torch.zeros(B, n_classes, 224, 224).cuda()
    for i in range(0, n_classes):
        for j in range(i + 1, n_classes):
            target_s2[:, i] += target[:, j]
        if i > 0:
            for z in range(0, i):
                target_s2[:, i] += target[:, z]
    return target_s2

def trainer_synapse(args, model, snapshot_path, root_path, list_dir, num_classes,w_glob,net_glob,net_previous):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    # from datasets.dataset_acdc import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    base_lr = args.base_lr
    num_classes = num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=root_path, list_dir=list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)     #gao

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    criterion = CrossEntropyLoss(reduction="none").cuda()
    ce_loss = CrossEntropyLoss()
    ce_w_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    s2_loss = S2_Loss(num_classes)

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    # base_optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = GAM(model.parameters(), base_optimizer=base_optimizer, model=model, args=args)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    u_list = []
    u_label_list = []
    for epoch_num in iterator:
        # A value
        # A = get_cov_mask(torch.zeros(768)).cuda()
        # select_ratio_MA = []
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # optimizer.set_closure(loss_fn, image_batch, label_batch)
            # outputs, loss = optimizer.step()

            outputs, afma_pred, att = model(image_batch)

            outputs_temp = outputs

            # afma_loss
            loss_afma = MyLoss_correction(att_depth=3, out_channels=num_classes, patch_size=7)
            loss_att = loss_afma(afma_pred, label_batch, att)

            # one_hot_label = _one_hot_encoder(label_batch, num_classes).float()

            loss_ce_s = ce_w_loss(afma_pred, label_batch[:])
            # loss_dice_s = dice_loss(afma_pred, label_batch, softmax=True)

            # stable
            # weight1 = weight_learner(cfeatures, x5, args, args.epochs, i_batch)
            # loss_sn_pre = torch.mean(criterion(outputs, label_batch[:].long()), dim=(1, 2), keepdim=False).view(1, -1)
            # loss_sn_pre = nn.Linear(outputs.shape[2]**2, 1, bias=True).cuda()(    #50176
            #     criterion(outputs, label_batch[:].long()).view(weight1.size()[0], -1)).view(1, -1)
            # loss_sn = loss_sn_pre.mm(weight1).view(1)
            # A, loss_A = SVI(loss_sn, outputs, label_batch, weight1, select_ratio_MA, args)
            # print(loss_sn)
            # label_clone = label_batch.clone()
            # s2_label = s2_change(label_clone, num_classes)
            # loss_s2 = s2_loss(outputs, s2_label, softmax=True)

            loss_ce = ce_loss(outputs, label_batch[:])
            loss_dice = dice_loss(outputs, label_batch, softmax=True)

            loss = 0.4 * loss_ce + 0.6 * loss_dice + 0.1 * loss_ce_s + loss_att

            if args.alg == 'FedProx':
                fed_prox_reg = 0.0
                # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
                for param_index, param in enumerate(model.parameters()):
                    fed_prox_reg += ((0.0001 / 2) * torch.norm((param - w_glob[param_index])) ** 2)
                loss += fed_prox_reg

            if args.alg == 'Moon':
                t = 0.5
                mu = 1.0
                outputs2, _, _ = net_glob(image_batch)
                outputs3, _, _ = net_previous(image_batch)

                cos = torch.nn.CosineSimilarity(dim=1)
                posi = cos(outputs, outputs2)/t
                nega = cos(outputs, outputs3)/t
                logits = -torch.log(torch.exp(posi)/(torch.add(torch.exp(posi), torch.exp(nega))))
                logits = mu * torch.mean(logits)

                loss += logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_sn', loss_sn, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # writer.add_scalar('info/loss_ce', loss, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            # print('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            # Uncer_thr

        #     net = model
        #     u, un_gt = pred_thresh(net, num_classes, outputs_temp, label_batch, image_batch.size(0))
        #     # u_list.append(list(np.array(u).flatten()))
        #     # u_label_list.append(list(np.array(un_gt).flatten()))
        #     u_list.append(u)
        #     u_label_list.append(un_gt)
        #
        # u_list = list(np.array(u_list).flatten())
        # u_label_list = list(np.array(u_label_list).flatten().astype('int'))
        #
        # fpr_Pri, tpr_Pri, thresh = metrics.roc_curve(u_label_list, u_list)
        # # tprs = TPR(u_label_list, u_list)
        # tnr_Pri = 1.0 - fpr_Pri
        # tpr_Pri = np.nan_to_num(tpr_Pri)
        # # max_j = max(zip(fpr_Pri, tpr_Pri), key=lambda x: 2 * x[1] - x[0])
        # max_j = max(zip(tnr_Pri, tpr_Pri), key=lambda x: x[1] + x[0] - 1.0) # Youden
        # # max_th[i_batch] = max_j
        # # thresh = thresh[list(zip(fpr_Pri, tpr_Pri)).index(max_j)]
        # thresh = thresh[list(zip(tnr_Pri, tpr_Pri)).index(max_j)]   # Youden
        # print("opt_pred ===== {}".format(thresh))
        # print(np.mean(max_th))

        # save_interval = 50  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #
        # if epoch_num >= max_epoch - 1:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     iterator.close()
        #     break

    writer.close()
    return #thresh ,image_batch, label_batch

def trainer_amos(args, model, snapshot_path, root_path, list_dir, num_classes,w_glob,net_glob,net_previous):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    # from datasets.dataset_acdc import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    base_lr = args.base_lr
    num_classes = num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=root_path, list_dir=list_dir, split="train_all",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)     #gao

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    criterion = CrossEntropyLoss(reduction="none").cuda()
    ce_loss = ce_amos(n_classes=16)
    ce_w_loss = ce_amos(n_classes=16)
    dice_loss = DiceLoss_amos(n_classes=16)
    s2_loss = S2_Loss(num_classes)

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    # base_optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = GAM(model.parameters(), base_optimizer=base_optimizer, model=model, args=args)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    u_list = []
    u_label_list = []
    for epoch_num in iterator:
        # A value
        # A = get_cov_mask(torch.zeros(768)).cuda()
        # select_ratio_MA = []
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # optimizer.set_closure(loss_fn, image_batch, label_batch)
            # outputs, loss = optimizer.step()

            outputs, afma_pred, att = model(image_batch)

            outputs_temp = outputs

            # afma_loss
            loss_afma = MyLoss_correction_amos(att_depth=3, out_channels=num_classes, patch_size=7)
            loss_att = loss_afma(afma_pred, label_batch, att)

            # one_hot_label = _one_hot_encoder(label_batch, num_classes).float()

            loss_ce_s = ce_w_loss(afma_pred, label_batch, softmax=True)
            # loss_dice_s = dice_loss(afma_pred, label_batch, softmax=True)

            # stable
            # weight1 = weight_learner(cfeatures, x5, args, args.epochs, i_batch)
            # loss_sn_pre = torch.mean(criterion(outputs, label_batch[:].long()), dim=(1, 2), keepdim=False).view(1, -1)
            # loss_sn_pre = nn.Linear(outputs.shape[2]**2, 1, bias=True).cuda()(    #50176
            #     criterion(outputs, label_batch[:].long()).view(weight1.size()[0], -1)).view(1, -1)
            # loss_sn = loss_sn_pre.mm(weight1).view(1)
            # A, loss_A = SVI(loss_sn, outputs, label_batch, weight1, select_ratio_MA, args)
            # print(loss_sn)
            # label_clone = label_batch.clone()
            # s2_label = s2_change(label_clone, num_classes)
            # loss_s2 = s2_loss(outputs, s2_label, softmax=True)

            loss_ce = ce_loss(outputs, label_batch, softmax=True)
            loss_dice = dice_loss(outputs, label_batch, softmax=True)

            loss = 0.4 * loss_ce + 0.6 * loss_dice + 0.1 * loss_ce_s + loss_att

            if args.alg == 'FedProx':
                fed_prox_reg = 0.0
                # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
                for param_index, param in enumerate(model.parameters()):
                    fed_prox_reg += ((0.0001 / 2) * torch.norm((param - w_glob[param_index])) ** 2)
                loss += fed_prox_reg
            if args.alg == 'Moon':
                t = 0.5
                mu = 1.0
                outputs2, _, _ = net_glob(image_batch)
                outputs3, _, _ = net_previous(image_batch)

                cos = torch.nn.CosineSimilarity(dim=1)
                posi = cos(outputs, outputs2) / t
                nega = cos(outputs, outputs3) / t
                logits = -torch.log(torch.exp(posi) / (torch.add(torch.exp(posi), torch.exp(nega))))
                logits = mu * torch.mean(logits)

                loss += logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_sn', loss_sn, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # writer.add_scalar('info/loss_ce', loss, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            # print('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            # Uncer_thr

        #     net = model
        #     u, un_gt = pred_thresh(net, num_classes, outputs_temp, label_batch, image_batch.size(0))
        #     # u_list.append(list(np.array(u).flatten()))
        #     # u_label_list.append(list(np.array(un_gt).flatten()))
        #     u_list.append(u)
        #     u_label_list.append(un_gt)
        #
        # u_list = list(np.array(u_list).flatten())
        # u_label_list = list(np.array(u_label_list).flatten().astype('int'))
        #
        # fpr_Pri, tpr_Pri, thresh = metrics.roc_curve(u_label_list, u_list)
        # # tprs = TPR(u_label_list, u_list)
        # tpr_Pri = np.nan_to_num(tpr_Pri)
        # tnr_Pri = 1.0 - fpr_Pri
        # # max_j = max(zip(fpr_Pri, tpr_Pri), key=lambda x: 2 * x[1] - x[0])   # UIOS
        # max_j = max(zip(tnr_Pri, tpr_Pri), key=lambda x: x[1] + x[0] - 1.0)
        # # max_th[i_batch] = max_j
        # thresh = thresh[list(zip(tnr_Pri, tpr_Pri)).index(max_j)]
        # print("opt_pred ===== {}".format(thresh))
        # print(np.mean(max_th))

        # save_interval = 50  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #
        # if epoch_num >= max_epoch - 1:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     iterator.close()
        #     break

    writer.close()
    return #thresh ,image_batch, label_batch

def trainer_flare(args, model, snapshot_path, root_path, list_dir, num_classes,w_glob,net_glob,net_previous):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    # from datasets.dataset_acdc import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    base_lr = args.base_lr
    num_classes = num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=root_path, list_dir=list_dir, split="train_all",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)     #gao

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    criterion = CrossEntropyLoss(reduction="none").cuda()
    ce_loss = ce_flare(n_classes=16)
    ce_w_loss = ce_flare(n_classes=16)
    dice_loss = DiceLoss_flare(n_classes=16)
    s2_loss = S2_Loss(num_classes)

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    # base_optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = GAM(model.parameters(), base_optimizer=base_optimizer, model=model, args=args)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    u_list = []
    u_label_list = []
    for epoch_num in iterator:
        # A value
        # A = get_cov_mask(torch.zeros(768)).cuda()
        # select_ratio_MA = []
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # optimizer.set_closure(loss_fn, image_batch, label_batch)
            # outputs, loss = optimizer.step()

            outputs, afma_pred, att = model(image_batch)

            outputs_temp = outputs

            # afma_loss
            loss_afma = MyLoss_correction_flare(att_depth=3, out_channels=num_classes, patch_size=7)
            loss_att = loss_afma(afma_pred, label_batch, att)

            # one_hot_label = _one_hot_encoder(label_batch, num_classes).float()

            loss_ce_s = ce_w_loss(afma_pred, label_batch, softmax=True)
            # loss_dice_s = dice_loss(afma_pred, label_batch, softmax=True)

            # stable
            # weight1 = weight_learner(cfeatures, x5, args, args.epochs, i_batch)
            # loss_sn_pre = torch.mean(criterion(outputs, label_batch[:].long()), dim=(1, 2), keepdim=False).view(1, -1)
            # loss_sn_pre = nn.Linear(outputs.shape[2]**2, 1, bias=True).cuda()(    #50176
            #     criterion(outputs, label_batch[:].long()).view(weight1.size()[0], -1)).view(1, -1)
            # loss_sn = loss_sn_pre.mm(weight1).view(1)
            # A, loss_A = SVI(loss_sn, outputs, label_batch, weight1, select_ratio_MA, args)
            # print(loss_sn)
            # label_clone = label_batch.clone()
            # s2_label = s2_change(label_clone, num_classes)
            # loss_s2 = s2_loss(outputs, s2_label, softmax=True)

            loss_ce = ce_loss(outputs, label_batch, softmax=True)
            loss_dice = dice_loss(outputs, label_batch, softmax=True)

            loss = 0.4 * loss_ce + 0.6 * loss_dice + 0.1 * loss_ce_s + loss_att

            if args.alg == 'FedProx':
                fed_prox_reg = 0.0
                # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
                for param_index, param in enumerate(model.parameters()):
                    fed_prox_reg += ((0.0001 / 2) * torch.norm((param - w_glob[param_index])) ** 2)
                loss += fed_prox_reg
            if args.alg == 'Moon':
                t = 0.5
                mu = 1.0
                outputs2, _, _ = net_glob(image_batch)
                outputs3, _, _ = net_previous(image_batch)

                cos = torch.nn.CosineSimilarity(dim=1)
                posi = cos(outputs, outputs2) / t
                nega = cos(outputs, outputs3) / t
                logits = -torch.log(torch.exp(posi) / (torch.add(torch.exp(posi), torch.exp(nega))))
                logits = mu * torch.mean(logits)

                loss += logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_sn', loss_sn, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # writer.add_scalar('info/loss_ce', loss, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            # print('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            # Uncer_thr

        #     net = model
        #     u, un_gt = pred_thresh(net, num_classes, outputs_temp, label_batch, image_batch.size(0))
        #     # u_list.append(list(np.array(u).flatten()))
        #     # u_label_list.append(list(np.array(un_gt).flatten()))
        #     u_list.append(u)
        #     u_label_list.append(un_gt)
        #
        # u_list = list(np.array(u_list).flatten())
        # u_label_list = list(np.array(u_label_list).flatten().astype('int'))
        #
        # fpr_Pri, tpr_Pri, thresh = metrics.roc_curve(u_label_list, u_list)
        # # tprs = TPR(u_label_list, u_list)
        # tpr_Pri = np.nan_to_num(tpr_Pri)
        # tnr_Pri = 1.0 - fpr_Pri
        # # max_j = max(zip(fpr_Pri, tpr_Pri), key=lambda x: 2 * x[1] - x[0])   # UIOS
        # max_j = max(zip(tnr_Pri, tpr_Pri), key=lambda x: x[1] + x[0] - 1.0)
        # # max_th[i_batch] = max_j
        # thresh = thresh[list(zip(tnr_Pri, tpr_Pri)).index(max_j)]
        # print("opt_pred ===== {}".format(thresh))
        # print(np.mean(max_th))

        # save_interval = 50  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #
        # if epoch_num >= max_epoch - 1:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     iterator.close()
        #     break

    writer.close()
    return #thresh ,image_batch, label_batch