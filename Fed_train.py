import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from Fed_trainer import trainer_synapse, trainer_amos, trainer_flare
# from Fed_baseline_trainer import trainer_synapse, trainer_amos, trainer_flare
from config import get_config
from fed_utils.FedAvg import FedAvg
from fed_utils.FedShapley import FedShapley
import copy
import sys
from tqdm import tqdm

data_name = ['Synapse', 'AMOS', 'FLARE22']
parser = argparse.ArgumentParser()
parser.add_argument('--root_path_sy', type=str,
                    default='../data/'+data_name[0], help='root dir for data')
parser.add_argument('--root_path_am', type=str,
                    default='F:/AMOS', help='root dir for data')
parser.add_argument('--root_path_fl', type=str,
                    default='F:/FLARE22', help='root dir for data')
parser.add_argument('--dataset_0', type=str,
                    default=data_name[0], help='experiment_name')
parser.add_argument('--dataset_1', type=str,
                    default=data_name[1], help='experiment_name')
parser.add_argument('--dataset_2', type=str,
                    default=data_name[2], help='experiment_name')
parser.add_argument('--list_dir_sy', type=str,
                    default='./lists/lists_'+data_name[0], help='list dir')
parser.add_argument('--list_dir_am', type=str,
                    default='./lists/lists_'+data_name[1], help='list dir')
parser.add_argument('--list_dir_fl', type=str,
                    default='./lists/lists_'+data_name[2], help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')   # 9
parser.add_argument('--output_dir', type=str,
                    default='./output/Moon_1',
                    help='output dir')
parser.add_argument("--alg", type=str, default='Moon')

parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=1, help='maximum epoch number to train')
parser.add_argument('--rounds', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str,
                    default= './configs/swin_tiny_patch4_window7_224_lite.yaml',    #requir=True gao
                    metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

# opt
parser.add_argument("--base_opt", default='SGD', type=str, help="")

parser.add_argument("--grad_beta_0", default=1., type=float, help="scale for g0")
parser.add_argument("--grad_beta_1", default=1., type=float, help="scale for g1")
parser.add_argument("--grad_beta_2", default=-1., type=float, help="scale for g2")
parser.add_argument("--grad_beta_3", default=1., type=float, help="scale for g3")

# parser.add_argument("--grad_rho_max", default=0.04, type=int, help="")
# parser.add_argument("--grad_rho_min", default=0.02, type=int, help="")
parser.add_argument("--grad_rho", default=0.02, type=int, help="")

# parser.add_argument("--grad_norm_rho_max", default=0.04, type=int, help="")
# parser.add_argument("--grad_norm_rho_min", default=0.02, type=int, help="")
parser.add_argument("--grad_norm_rho", default=0.2, type=int, help="")

parser.add_argument("--adaptive", default=False, type=bool, help="")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")

parser.add_argument("--grad_gamma", default=0.03, type=int, help="")

# stable
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument ('--lrbl', type = float, default = 1.0, help = 'learning rate of balance')     # 1.0

parser.add_argument('--cos', '--cosine_lr', default=1, type=int,
                    metavar='COS', help='lr decay by decay', dest='cos')
parser.add_argument ('--epochb', type = int, default = 20, help = 'number of epochs to balance')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument ('--epochs_decay', type=list, default=[24, 30], help = 'weight lambda for second order moment loss')

parser.add_argument ('--num_f', type=int, default=1, help = 'number of fourier spaces')    # num_f 5 is best in paper, org is 1
parser.add_argument('--sum', type=bool, default=True, help='sum or concat')     # True

parser.add_argument ('--decay_pow', type=float, default=2, help = 'value of pow for weight decay')
# for expectation
parser.add_argument ('--lambda_decay_rate', type=float, default=1, help = 'ratio of epoch for lambda to decay')
parser.add_argument ('--lambda_decay_epoch', type=int, default=5, help = 'number of epoch for lambda to decay')
parser.add_argument ('--min_lambda_times', type=float, default=0.01, help = 'number of global table levels')

parser.add_argument ('--lambdap', type = float, default = 70.0, help = 'weight decay for weight1 ')

parser.add_argument ('--first_step_cons', type=float, default=1, help = 'constrain the weight at the first step')

parser.add_argument('--presave_ratio', type=float, default=0.9, help='the ratio for presaving features')

parser.add_argument("--period_MA", type=int, default=3)
parser.add_argument("--lam_STG", type=float, default=3)
parser.add_argument("--sigma_STG", type=float, default=0.1)
parser.add_argument("--a_dim", type=float, default=768)

args = parser.parse_args()
if args.dataset_0 == "Synapse":
    args.root_path_sy = os.path.join(args.root_path_sy, "train_npz")
if args.dataset_0 == "ACDC":
    args.root_path = os.path.join(args.root_path, "train")
config = get_config(args)
if args.dataset_1 == "AMOS":
    args.root_path_am = os.path.join(args.root_path_am, "train_npz_new")
if args.dataset_2 == "FLARE22":
    args.root_path_fl = os.path.join(args.root_path_fl, "train")


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name_0 = args.dataset_0
    dataset_name_1 = args.dataset_1
    dataset_name_2 = args.dataset_2

    dataset_config = {
        'Synapse': {
            'root_path': args.root_path_sy,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'ACDC': {
            'root_path': args.root_path_sy,  # ../data/Synapse/train_npz
            'list_dir': './lists/lists_ACDC',  # ./lists/lists_Synapse
            'num_classes': 4,
        },
        'AMOS': {
            'root_path': args.root_path_am,
            'list_dir': './lists/lists_AMOS',
            'num_classes': 16,    # 16
            # 'train_eg_dir': args.train_eg_dir,  # edge gao
        },
        'FLARE22': {
            'root_path': args.root_path_fl,
            'list_dir': './lists/lists_FLARE22',
            'num_classes': 14,  # 14
            # 'train_eg_dir': args.train_eg_dir,  # edge gao
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes_0 = dataset_config[dataset_name_0]['num_classes']
    args.root_path_0 = dataset_config[dataset_name_0]['root_path']
    args.list_dir_0 = dataset_config[dataset_name_0]['list_dir']

    args.num_classes_1 = dataset_config[dataset_name_0]['num_classes']      # for universal
    args.root_path_1 = dataset_config[dataset_name_1]['root_path']
    args.list_dir_1 = dataset_config[dataset_name_1]['list_dir']

    args.num_classes_2 = dataset_config[dataset_name_0]['num_classes']  # for universal
    args.root_path_2 = dataset_config[dataset_name_2]['root_path']
    args.list_dir_2 = dataset_config[dataset_name_2]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net_sy = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net_sy.load_from(config)

    net_am = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net_am.load_from(config)

    net_fl = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net_fl.load_from(config)

    # Resume = True       # 中断重启
    Resume = False
    if Resume:
        path_checkpoint = 'D:\Project_fan\Swin-AFMA\output/Ablation/FedUC_gamma_0.6/epoch_49.pth'
        checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'))
        net_sy.load_state_dict(checkpoint)
        net_am.load_state_dict(checkpoint)
        net_fl.load_state_dict(checkpoint)

    # FedAvg for synapse and amos
    net_glob = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net_glob.load_from(config)
    net_glob.eval()
    for param in net_glob.parameters():
        param.requires_grad = False
    w_glob = net_glob.state_dict()
    param_g = list(net_glob.cuda().parameters())
    w_locals = []

    # if args.alg == 'Moon':
    #     net_previous = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    #     net_previous.load_from(config)
    #     net_previous.eval()
    #     for param in net_previous.parameters():
    #         param.requires_grad = False
        # w_previous = copy.deepcopy(w_glob)
        # net_previous.load_state_dict(w_previous)

    for id in range(3):
        w_locals.append(copy.deepcopy(w_glob))

    dict_len = [932.0, 932.0, 932.0]

    logging.basicConfig(filename=args.output_dir + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    rounds = args.rounds
    if Resume:
        rounds = 100

    iterator = tqdm(range(rounds), ncols=70)
    # global_shapley = []

    for round in iterator:

        logging.info("round:{}".format(round))

        Fed_trainer_0 = {args.dataset_0: trainer_synapse,}
        thr0, img_0, label_0 = Fed_trainer_0[dataset_name_0](args, net_sy, args.output_dir, args.root_path_sy, args.list_dir_sy, args.num_classes_0)
        # Fed_trainer_0[dataset_name_0](args, net_sy, args.output_dir, args.root_path_sy,
        #                                                      args.list_dir_sy, args.num_classes_0,param_g,net_glob,net_previous)
        w_0 = net_sy.state_dict()
        w_locals[0] = copy.deepcopy(w_0)

        Fed_trainer_1 = {args.dataset_1: trainer_amos, }
        thr1, img_1, label_1 = Fed_trainer_1[dataset_name_1](args, net_am, args.output_dir, args.root_path_1, args.list_dir_1, args.num_classes_1)
        # Fed_trainer_1[dataset_name_1](args, net_am, args.output_dir, args.root_path_1,
        #                                                      args.list_dir_1, args.num_classes_1,param_g,net_glob,net_previous)
        w_1 = net_am.state_dict()
        w_locals[1] = copy.deepcopy(w_1)

        Fed_trainer_2 = {args.dataset_2: trainer_flare, }
        thr2, img_2, label_2 = Fed_trainer_2[dataset_name_2](args, net_fl, args.output_dir, args.root_path_2, args.list_dir_2,
                                             args.num_classes_2)
        # Fed_trainer_2[dataset_name_2](args, net_fl, args.output_dir, args.root_path_2, args.list_dir_2,
        #                                                args.num_classes_2,param_g,net_glob,net_previous)
        w_2 = net_fl.state_dict()
        w_locals[2] = copy.deepcopy(w_2)

        # data for all client
        img = [img_0, img_1, img_2]
        label = [label_0, label_1, label_2]

        # weight for all client
        thr0 = torch.exp(torch.tensor(thr0))
        thr1 = torch.exp(torch.tensor(thr1))
        thr2 = torch.exp(torch.tensor(thr2))
        thr = torch.tensor([thr0, thr1, thr2])
        thr = torch.softmax(thr, 0)
        thr_w = list(thr.flatten())

        thr_w.append(torch.softmax(thr, 1))

        # upload and download
        # if round == 0:
        #     # l = torch.tensor([2211.0, 26069.0, 4794.0])
        #     l = torch.tensor([2211.0, 2211.0, 2211.0])  # Avg
        #     # l = [2211.0, 2321.0]
        #     for ele in l:
        #         global_shapley.append(ele / sum(l))

        # if args.alg == 'Moon':
        #     if round > 1:
        #         w_previous = copy.deepcopy(w_glob)
        #         net_previous.load_state_dict(w_previous)

        with torch.no_grad():
            # w_glob = FedAvg(w_locals, dict_len)
            w_glob = FedShapley(w_locals, net_glob, round, img, label, thr_w)

        net_glob.load_state_dict(w_glob)  # Moon

        net_sy.load_state_dict(w_glob)
        net_am.load_state_dict(w_glob)
        net_fl.load_state_dict(w_glob)

        save_interval = 50  # int(max_epoch/6)
        if Resume == False:
            if round == 49:
                save_mode_path = os.path.join(args.output_dir, 'epoch_' + str(round) + '.pth')
                torch.save(w_glob, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if round > int(rounds / 2) and (round + 1) % save_interval == 0:
                save_mode_path = os.path.join(args.output_dir, 'epoch_' + str(round) + '.pth')
                torch.save(w_glob, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if round >= rounds - 1:
                save_mode_path = os.path.join(args.output_dir, 'epoch_' + str(round) + '.pth')
                torch.save(w_glob, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                iterator.close()
                break
        else:
            if round == 49:
                save_mode_path = os.path.join(args.output_dir, 'epoch_' + str(round+50) + '.pth')
                torch.save(w_glob, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if round >= rounds - 1:
                save_mode_path = os.path.join(args.output_dir, 'epoch_' + str(round+50) + '.pth')
                torch.save(w_glob, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                iterator.close()
                break


    print("Training Finished!")