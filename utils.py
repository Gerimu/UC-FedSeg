import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from base import Loss

# import matplotlib.pyplot as plt
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2

def add_gaussian_noise(img: torch.Tensor, sigma: float = 0.05):
    """
    在保持 input 原值基础上加高斯噪声
    img: 任意范围，不做 clip
    sigma: 噪声标准差，相对于图像取值范围自行设定
    """
    noise = torch.randn_like(img) * sigma
    return img + noise

def add_poisson_noise(img: torch.Tensor, lam_scale: float = 0.1):
    """
    在保持 input 原值基础上加泊松噪声
    注意：Poisson 分布要求输入非负，否则报错
    lam_scale: 越大噪声越小
    """ 
    img01 = (img + 1) / 2.0         # [-1,1] → [0,1]
    lam = img01 * lam_scale
    noisy01 = torch.poisson(lam) / lam_scale
    noisy01 = torch.clamp(noisy01, 0.0, 1.0)
    return noisy01 * 2.0 - 1.0      # [0,1] → [-1,1]

def get_cov_mask(select_ratio):
    select_ratio = torch.reshape(select_ratio, (-1, 1))
    A = 1-torch.matmul(select_ratio, select_ratio.T)
    return A

class MyLoss_correction(Loss):
    def __init__(self,weight=None,att_depth=None,out_channels=None,patch_size=None):
        super().__init__()
        self.nll = nn.NLLLoss(weight=weight)
        self.mseloss=nn.MSELoss()

        self.att_depth=att_depth
        self.patch_size=patch_size
        self.out_channels= out_channels
        self.num_classes = out_channels

        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                    stride=(self.patch_size, self.patch_size))

    def _one_hot_encoder(self, input_tensor, num_classes):
        tensor_list = []
        for i in range(num_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, y_pr, y_gt, attentions):
        conv_feamap_size = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(2 ** self.att_depth, 2 ** self.att_depth),
                             stride=(2 ** self.att_depth, 2 ** self.att_depth), groups=self.out_channels, bias=False)
        conv_feamap_size.weight = nn.Parameter(torch.ones((self.out_channels, 1, 2 ** self.att_depth, 2 ** self.att_depth)))
        conv_feamap_size.to(y_pr.device)
        for param in conv_feamap_size.parameters():
            param.requires_grad = False

        y_gt = self._one_hot_encoder(y_gt, num_classes=self.num_classes)

        y_gt_conv=conv_feamap_size(y_gt)/(2 ** self.att_depth*2 ** self.att_depth)

        attentions_gt=[]

        for i in range(y_gt_conv.size()[1]):
            unfold_y_gt = self.unfold(y_gt[:, i:i + 1, :, :]).transpose(-1, -2)
            unfold_y_gt_conv = self.unfold(y_gt_conv[:, i:i + 1, :, :])
            att=torch.matmul(unfold_y_gt,unfold_y_gt_conv)/(self.patch_size*self.patch_size)
            att=torch.unsqueeze(att,dim=1)
            attentions_gt.append(att)

        attentions_gt=torch.cat(attentions_gt,dim=1)

        # y_gt=torch.argmax(y_gt,dim=-3)
        # y_pr=torch.softmax(y_pr,dim=1)
        #
        # loss_entropy=self.nll(y_pr,y_gt)
        loss_mse = self.mseloss(attentions, attentions_gt)

        loss=loss_mse

        return loss

class MyLoss_correction_amos(Loss):
    def __init__(self,weight=None,att_depth=None,out_channels=None,patch_size=None):
        super().__init__()
        self.nll = nn.NLLLoss(weight=weight)
        self.mseloss=nn.MSELoss()

        self.att_depth=att_depth
        self.patch_size=patch_size
        self.out_channels= 16
        self.num_classes = 16

        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                    stride=(self.patch_size, self.patch_size))

    def _one_hot_encoder(self, input_tensor, num_classes):
        tensor_list = []
        for i in range(num_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _mse_loss(self, score, target):
        sq_error = (score - target) ** 2
        return sq_error.view(-1)

    def forward(self, y_pr, y_gt, attentions):
        conv_feamap_size = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(2 ** self.att_depth, 2 ** self.att_depth),
                             stride=(2 ** self.att_depth, 2 ** self.att_depth), groups=self.out_channels, bias=False)
        conv_feamap_size.weight = nn.Parameter(torch.ones((self.out_channels, 1, 2 ** self.att_depth, 2 ** self.att_depth)))
        conv_feamap_size.to(y_pr.device)
        for param in conv_feamap_size.parameters():
            param.requires_grad = False

        y_gt = self._one_hot_encoder(y_gt, num_classes=self.num_classes)

        y_gt_conv=conv_feamap_size(y_gt)/(2 ** self.att_depth*2 ** self.att_depth)

        attentions_gt=[]

        for i in range(y_gt_conv.size()[1]):
            unfold_y_gt = self.unfold(y_gt[:, i:i + 1, :, :]).transpose(-1, -2)
            unfold_y_gt_conv = self.unfold(y_gt_conv[:, i:i + 1, :, :])
            att=torch.matmul(unfold_y_gt,unfold_y_gt_conv)/(self.patch_size*self.patch_size)
            att=torch.unsqueeze(att,dim=1)
            attentions_gt.append(att)

        attentions_gt=torch.cat(attentions_gt,dim=1)

        # y_gt=torch.argmax(y_gt,dim=-3)
        # y_pr=torch.softmax(y_pr,dim=1)
        #
        # loss_entropy=self.nll(y_pr,y_gt)
        mselist = []
        mselist.append(self._mse_loss(attentions[:, 0], attentions_gt[:, 0]))
        mselist.append(self._mse_loss(attentions[:, 1], attentions_gt[:, 8]))
        mselist.append(self._mse_loss(attentions[:, 2], attentions_gt[:, 4]))
        mselist.append(self._mse_loss(attentions[:, 3], attentions_gt[:, 3]))
        mselist.append(self._mse_loss(attentions[:, 4], attentions_gt[:, 2]))
        mselist.append(self._mse_loss(attentions[:, 5], attentions_gt[:, 6]))
        mselist.append(self._mse_loss(attentions[:, 6], attentions_gt[:, 10]))
        mselist.append(self._mse_loss(attentions[:, 7], attentions_gt[:, 1]))
        mselist.append(self._mse_loss(attentions[:, 8], attentions_gt[:, 7]))
        # loss_mse = self.mseloss(attentions, attentions_gt)
        loss_mse = sum(mselist)

        loss = torch.sum((loss_mse / self.num_classes),dim=0, keepdim=False) / loss_mse.size(0)

        return loss

class MyLoss_correction_flare(Loss):
    def __init__(self,weight=None,att_depth=None,out_channels=None,patch_size=None):
        super().__init__()
        self.nll = nn.NLLLoss(weight=weight)
        self.mseloss=nn.MSELoss()

        self.att_depth=att_depth
        self.patch_size=patch_size
        self.out_channels= 16
        self.num_classes = 16

        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                    stride=(self.patch_size, self.patch_size))

    def _one_hot_encoder(self, input_tensor, num_classes):
        tensor_list = []
        for i in range(num_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _mse_loss(self, score, target):
        sq_error = (score - target) ** 2
        return sq_error.view(-1)

    def forward(self, y_pr, y_gt, attentions):
        conv_feamap_size = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(2 ** self.att_depth, 2 ** self.att_depth),
                             stride=(2 ** self.att_depth, 2 ** self.att_depth), groups=self.out_channels, bias=False)
        conv_feamap_size.weight = nn.Parameter(torch.ones((self.out_channels, 1, 2 ** self.att_depth, 2 ** self.att_depth)))
        conv_feamap_size.to(y_pr.device)
        for param in conv_feamap_size.parameters():
            param.requires_grad = False

        y_gt = self._one_hot_encoder(y_gt, num_classes=self.num_classes)

        y_gt_conv=conv_feamap_size(y_gt)/(2 ** self.att_depth*2 ** self.att_depth)

        attentions_gt=[]

        for i in range(y_gt_conv.size()[1]):
            unfold_y_gt = self.unfold(y_gt[:, i:i + 1, :, :]).transpose(-1, -2)
            unfold_y_gt_conv = self.unfold(y_gt_conv[:, i:i + 1, :, :])
            att=torch.matmul(unfold_y_gt,unfold_y_gt_conv)/(self.patch_size*self.patch_size)
            att=torch.unsqueeze(att,dim=1)
            attentions_gt.append(att)

        attentions_gt=torch.cat(attentions_gt,dim=1)

        # y_gt=torch.argmax(y_gt,dim=-3)
        # y_pr=torch.softmax(y_pr,dim=1)
        #
        # loss_entropy=self.nll(y_pr,y_gt)
        mselist = []
        mselist.append(self._mse_loss(attentions[:, 0], attentions_gt[:, 0]))
        mselist.append(self._mse_loss(attentions[:, 1], attentions_gt[:, 5]))
        mselist.append(self._mse_loss(attentions[:, 2], attentions_gt[:, 9]))
        mselist.append(self._mse_loss(attentions[:, 3], attentions_gt[:, 13]))
        mselist.append(self._mse_loss(attentions[:, 4], attentions_gt[:, 2]))
        mselist.append(self._mse_loss(attentions[:, 5], attentions_gt[:, 1]))
        mselist.append(self._mse_loss(attentions[:, 6], attentions_gt[:, 4]))
        mselist.append(self._mse_loss(attentions[:, 7], attentions_gt[:, 3]))
        mselist.append(self._mse_loss(attentions[:, 8], attentions_gt[:, 11]))
        # loss_mse = self.mseloss(attentions, attentions_gt)
        loss_mse = sum(mselist)

        loss = torch.sum((loss_mse / self.num_classes),dim=0, keepdim=False) / loss_mse.size(0)

        return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class S2_Loss(nn.Module):
    def __init__(self, n_classes):
        super(S2_Loss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        # loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class DiceLoss_amos(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss_amos, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss.view(-1)

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        # assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        dicelist = []
        loss = 0.0
        # for i in range(0, self.n_classes):
        dicelist.append(self._dice_loss(inputs[:, 0], target[:, 0]))
        dicelist.append(self._dice_loss(inputs[:, 1], target[:, 8]))
        dicelist.append(self._dice_loss(inputs[:, 2], target[:, 4]))
        dicelist.append(self._dice_loss(inputs[:, 3], target[:, 3]))
        dicelist.append(self._dice_loss(inputs[:, 4], target[:, 2]))
        dicelist.append(self._dice_loss(inputs[:, 5], target[:, 6]))
        dicelist.append(self._dice_loss(inputs[:, 6], target[:, 10]))
        dicelist.append(self._dice_loss(inputs[:, 7], target[:, 1]))
        dicelist.append(self._dice_loss(inputs[:, 8], target[:, 7]))

        # class_wise_dice.append(1.0 - dice.item())
        loss = sum(dicelist)
        loss = torch.sum((loss / 9), dim=0, keepdim=False) / loss.size(0)
        return loss

class DiceLoss_flare(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss_flare, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss.view(-1)

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        # assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        dicelist = []
        loss = 0.0
        # for i in range(0, self.n_classes):
        dicelist.append(self._dice_loss(inputs[:, 0], target[:, 0]))
        dicelist.append(self._dice_loss(inputs[:, 1], target[:, 5]))
        dicelist.append(self._dice_loss(inputs[:, 2], target[:, 9]))
        dicelist.append(self._dice_loss(inputs[:, 3], target[:, 13]))
        dicelist.append(self._dice_loss(inputs[:, 4], target[:, 2]))
        dicelist.append(self._dice_loss(inputs[:, 5], target[:, 1]))
        dicelist.append(self._dice_loss(inputs[:, 6], target[:, 4]))
        dicelist.append(self._dice_loss(inputs[:, 7], target[:, 3]))
        dicelist.append(self._dice_loss(inputs[:, 8], target[:, 11]))

        # class_wise_dice.append(1.0 - dice.item())
        loss = sum(dicelist)
        loss = torch.sum((loss / 9), dim=0, keepdim=False) / loss.size(0)
        return loss

class ce_amos(nn.Module):
    def __init__(self, n_classes):
        super(ce_amos, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _ce_loss(self, score, target):
        loss = -(target * torch.log(score))
        return loss.view(-1)

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        # assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        dicelist = []
        loss = 0.0
        # for i in range(0, self.n_classes):
        dicelist.append(self._ce_loss(inputs[:, 0], target[:, 0]))
        dicelist.append(self._ce_loss(inputs[:, 1], target[:, 8]))
        dicelist.append(self._ce_loss(inputs[:, 2], target[:, 4]))
        dicelist.append(self._ce_loss(inputs[:, 3], target[:, 3]))
        dicelist.append(self._ce_loss(inputs[:, 4], target[:, 2]))
        dicelist.append(self._ce_loss(inputs[:, 5], target[:, 6]))
        dicelist.append(self._ce_loss(inputs[:, 6], target[:, 10]))
        dicelist.append(self._ce_loss(inputs[:, 7], target[:, 1]))
        dicelist.append(self._ce_loss(inputs[:, 8], target[:, 7]))

        # class_wise_dice.append(1.0 - dice.item())
        loss = sum(dicelist)
        loss = torch.sum(loss , dim=0, keepdim=False) / loss.size(0)
        return loss

class ce_flare(nn.Module):
    def __init__(self, n_classes):
        super(ce_flare, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _ce_loss(self, score, target):
        loss = -(target * torch.log(score))
        return loss.view(-1)

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        # assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        dicelist = []
        loss = 0.0
        # for i in range(0, self.n_classes):
        dicelist.append(self._ce_loss(inputs[:, 0], target[:, 0]))
        dicelist.append(self._ce_loss(inputs[:, 1], target[:, 5]))
        dicelist.append(self._ce_loss(inputs[:, 2], target[:, 9]))
        dicelist.append(self._ce_loss(inputs[:, 3], target[:, 13]))
        dicelist.append(self._ce_loss(inputs[:, 4], target[:, 2]))
        dicelist.append(self._ce_loss(inputs[:, 5], target[:, 1]))
        dicelist.append(self._ce_loss(inputs[:, 6], target[:, 4]))
        dicelist.append(self._ce_loss(inputs[:, 7], target[:, 3]))
        dicelist.append(self._ce_loss(inputs[:, 8], target[:, 11]))

        # class_wise_dice.append(1.0 - dice.item())
        loss = sum(dicelist)
        loss = torch.sum(loss , dim=0, keepdim=False) / loss.size(0)
        return loss

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt)
        hd95 = 0
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def mask_classes(mask, num_classes = 3):
    _mask = [mask == i for i in range(1, num_classes + 1)]
    # _mask = torch.tensor([item.cpu().detach().numpy() for item in _mask]).cuda()
    # _mask = _mask.cpu().numpy().astype(np.uint8)
    class_rv = np.array(_mask[0].cpu()).astype(np.uint8)
    class_myo = np.array(_mask[1].cpu()).astype(np.uint8)
    class_lv = np.array(_mask[2].cpu()).astype(np.uint8)
    # class_xx = np.array(_mask[3].cpu()).astype(np.uint8)
    # img = torch.from_numpy(class_lv.squeeze(0)).permute(1, 2, 0)
    # img = img.detach().numpy()
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img = img.astype(np.float32)
    # plt.imshow(img)
    # plt.show()
    return class_rv, class_myo, class_lv

def reshape_transform(tensor, height=224, width=224):
    '''
    不同参数的Swin网络的height和width是不同的，具体需要查看所对应的配置文件yaml
    height = width = config.DATA.IMG_SIZE / config.MODEL.NUM_HEADS[-1]
    比如该例子中IMG_SIZE: 224  NUM_HEADS: [4, 8, 16, 32]
    height = width = 224 / 32 = 7
    '''
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# class SemanticSegmentationTarget:
#     def __init__(self, category, mask):
#         self.category = category
#         self.mask = torch.from_numpy(mask)
#         if torch.cuda.is_available():
#             self.mask = self.mask.cuda()
#
#     def __call__(self, model_output):
#         # output_model = [model_output == i for i in range(1, 4)]
#         # class_rv = np.array(output_model[0].cpu()).astype(np.uint8)
#         # class_myo = np.array(output_model[1].cpu()).astype(np.uint8)
#         # class_lv = np.array(output_model[2].cpu()).astype(np.uint8)
#         # img = torch.from_numpy(class_rv).permute(1, 2, 0)
#         # img = img.detach().numpy()
#         # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#         # img = img.astype(np.float32)
#         # plt.imshow(img)
#         # plt.show()
#         # model_output = self.model_output.squeeze(0)
#
#         return (model_output[self.category, :, :] * self.mask).sum()

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    metric_all_slice = []
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]          # slice
            slice_label = label[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()

            # # CAM gao 1
            #
            # target_layers = [net.swin_unet.up.norm]
            # cam = GradCAM(model=net, target_layers=target_layers,
            #               use_cuda=True, reshape_transform=reshape_transform)
            # # CAM gao 1
            # # slice_label = label[ind, :, :]  # label slice
            # # x, y = slice_label.shape[0], slice_label.shape[1]
            # # if x != patch_size[0] or y != patch_size[1]:
            # #     slice_label = zoom(slice_label, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            # # label_three = torch.from_numpy(slice_label).unsqueeze(0).unsqueeze(0).float().cuda()
            # # rv, myo, lv = mask_classes(label_three)
            # # cam_label = [rv.squeeze(0), myo.squeeze(0), lv.squeeze(0)]
            # # CAM gao
            # input_tensor = torch.from_numpy(slice).unsqueeze(2).float()
            # input_tensor = input_tensor.numpy()
            # # input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_GRAY2RGB)
            # input_tensor = np.concatenate((input_tensor, input_tensor, input_tensor), axis=-1)
            # input_tensor = input_tensor * 255
            # input_tensor = cv2.normalize(input_tensor, None, 0, 255, cv2.NORM_MINMAX)
            # # input_tensor = np.array(input_tensor, dtype='uint8')
            # input_tensor = input_tensor.astype(np.uint8) / 255
            # # down = nn.Linear(1, 3)
            # # image2 = input_tensor.squeeze(0).cpu().permute(1, 2, 0)         # GRAY2RGB
            # # image2 = image2.detach().numpy()
            # # image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
            # # image2 = image2.astype(np.float32) / 255
            # # plt.imshow(input_tensor)
            # # plt.show()
            #
            # # input_tensor2 = preprocess_image(input_tensor, mean=[0.5, 0.5, 0.5],
            # #                                 std=[0.5, 0.5, 0.5])
            # mask_test = net(input)
            # normalized_masks = torch.nn.functional.softmax(mask_test, dim=1)  # .cpu
            # # normalized_masks = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            # sem_classes = [
            #     '__background__', 'Aorta', 'Gallbladder', 'Kidney(L)', 'Kidney(R)', 'Liver', 'Pancreas', 'Spleen', 'Stomach'
            # ]
            # # sem_classes = [
            # #     '__background__', 'RV', 'Myo', 'LV'
            # # ]
            # sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
            #
            # car_category = sem_class_to_idx["Aorta"]
            # car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
            # car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
            # car_mask_float = np.float32(car_mask == car_category)
            #
            # cam.batch_size = 1
            #
            # # class_map = {0:'Background', 1: 'RV', 2: 'Myo', 3: 'LV'}
            # class_map = {0:'Background', 1: 'Aorta', 2: 'Gallbladder', 3: 'Kidney(L)', 4: 'Kidney(R)', 5: 'Liver', 6: 'Pancreas', 7: 'Spleen', 8: 'Stomach'}
            # save_path = "D:\Project_fan\Swin-Unet-main\plot"
            # # [SemanticSegmentationTarget(class_id, mask=cam_label[class_id - 1])]
            # for i in range(9):
            #     class_id = i
            #     class_name = class_map[class_id]
            #     targets = [SemanticSegmentationTarget(i, car_mask_float)]
            #
            #     grayscale_cam = cam(input_tensor=input,
            #                         targets=targets,
            #                         eigen_smooth=False,
            #                         aug_smooth=False
            #                         )
            #
            #     # Here grayscale_cam has only one image in the batch
            #     grayscale_cam = grayscale_cam[0, :]
            #
            #     cam_image = show_cam_on_image(input_tensor, grayscale_cam, use_rgb=True)
            #     plt.imshow(cam_image)
            #     # plt.title(class_name)
            #     plt.axis('off')  # 去坐标轴
            #     plt.xticks([])  # 去 x 轴刻度
            #     plt.yticks([])  # 去 y 轴刻度
            #     plt.savefig(save_path+"/swin_slice_100_{}".format(i), bbox_inches='tight',pad_inches=0)
            #     plt.show()
            #     # CAM gao
            
            input = add_gaussian_noise(input, sigma=0.1)
            # input = add_poisson_noise(input, lam_scale=50) 

            with torch.no_grad():
                outputs,_,_ = net(input)
                # outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
                # p = []
                # for i in range(1, classes):
                #     p.append(calculate_metric_percase(pred == i, slice_label == i))
                # performance = np.mean(p, axis=0)[0]
                # mean_hd95 = np.mean(p, axis=0)[1]
                # metric_all_slice.append([performance, mean_hd95])  # gao
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    # p = []
    # p.append(calculate_metric_percase(prediction == 6, label == 6))
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # metric_list.append(calculate_metric_percase(prediction == 1, label == 8))
    # metric_list.append(calculate_metric_percase(prediction == 2, label == 4))
    # metric_list.append(calculate_metric_percase(prediction == 3, label == 3))
    # metric_list.append(calculate_metric_percase(prediction == 4, label == 2))
    # metric_list.append(calculate_metric_percase(prediction == 5, label == 6))
    # metric_list.append(calculate_metric_percase(prediction == 6, label == 10))
    # metric_list.append(calculate_metric_percase(prediction == 7, label == 1))
    # metric_list.append(calculate_metric_percase(prediction == 8, label == 7))

    # metric_list.append(calculate_metric_percase(prediction == 1, label == 5))
    # metric_list.append(calculate_metric_percase(prediction == 2, label == 9))
    # metric_list.append(calculate_metric_percase(prediction == 3, label == 13))
    # metric_list.append(calculate_metric_percase(prediction == 4, label == 2))
    # metric_list.append(calculate_metric_percase(prediction == 5, label == 1))
    # metric_list.append(calculate_metric_percase(prediction == 6, label == 4))
    # metric_list.append(calculate_metric_percase(prediction == 7, label == 3))
    # metric_list.append(calculate_metric_percase(prediction == 8, label == 11))

    # metric_list.append(calculate_metric_percase(prediction == 1, label == 8))
    # metric_list.append(calculate_metric_percase(prediction == 2, label == 6))
    # metric_list.append(calculate_metric_percase(prediction == 3, label == 3))
    # metric_list.append(calculate_metric_percase(prediction == 4, label == 4))
    # metric_list.append(calculate_metric_percase(prediction == 5, label == 1))
    # metric_list.append(calculate_metric_percase(prediction == 6, label == 8))
    # metric_list.append(calculate_metric_percase(prediction == 7, label == 2))
    # metric_list.append(calculate_metric_percase(prediction == 8, label == 5))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def test_single_volume_amos(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    metric_all_slice = []
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]          # slice
            slice_label = label[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()

            with torch.no_grad():
                outputs,_,_ = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
                # p = []
                # for i in range(1, classes):
                #     p.append(calculate_metric_percase(pred == i, slice_label == i))
                # performance = np.mean(p, axis=0)[0]
                # mean_hd95 = np.mean(p, axis=0)[1]
                # metric_all_slice.append([performance, mean_hd95])  # gao
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    # p = []
    # p.append(calculate_metric_percase(prediction == 6, label == 6))
    # for i in range(1, classes):
    #     metric_list.append(calculate_metric_percase(prediction == i, label == i))

    metric_list.append(calculate_metric_percase(prediction == 1, label == 8))
    metric_list.append(calculate_metric_percase(prediction == 2, label == 4))
    metric_list.append(calculate_metric_percase(prediction == 3, label == 3))
    metric_list.append(calculate_metric_percase(prediction == 4, label == 2))
    metric_list.append(calculate_metric_percase(prediction == 5, label == 6))
    metric_list.append(calculate_metric_percase(prediction == 6, label == 10))
    metric_list.append(calculate_metric_percase(prediction == 7, label == 1))
    metric_list.append(calculate_metric_percase(prediction == 8, label == 7))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def test_single_volume_flare(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    metric_all_slice = []
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]          # slice
            slice_label = label[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()

            with torch.no_grad():
                outputs,_,_ = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
                # p = []
                # for i in range(1, classes):
                #     p.append(calculate_metric_percase(pred == i, slice_label == i))
                # performance = np.mean(p, axis=0)[0]
                # mean_hd95 = np.mean(p, axis=0)[1]
                # metric_all_slice.append([performance, mean_hd95])  # gao
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    # p = []
    # p.append(calculate_metric_percase(prediction == 6, label == 6))
    # for i in range(1, classes):
    #     metric_list.append(calculate_metric_percase(prediction == i, label == i))
    metric_list.append(calculate_metric_percase(prediction == 1, label == 5))
    metric_list.append(calculate_metric_percase(prediction == 2, label == 9))
    metric_list.append(calculate_metric_percase(prediction == 3, label == 13))
    metric_list.append(calculate_metric_percase(prediction == 4, label == 2))
    metric_list.append(calculate_metric_percase(prediction == 5, label == 1))
    metric_list.append(calculate_metric_percase(prediction == 6, label == 4))
    metric_list.append(calculate_metric_percase(prediction == 7, label == 3))
    metric_list.append(calculate_metric_percase(prediction == 8, label == 11))


    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list