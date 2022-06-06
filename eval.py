import torch
import numpy as np


class EvalMetrics:
    def __init__(self, n_classes=4):
        # self.pred = torch.reshape(pred, actual.shape)
        # self.actual = actual
        self.n_classes = n_classes

    # def mask2onehot(self, mask):
    #     """
    #     Converts a segmentation mask (N,H,W) to (K,N,H,W) where K is the number of classes,
    #     N is the number of images, H is the number of rows and W is the number of columns
    #     """
    #     N, H, W = mask.shape
    #     _mask = torch.zeros([self.n_classes, N, H, W])
    #     for i in range(self.n_classes):
    #         _mask[i][mask == i] = 1
    #     _mask = _mask.permute(1, 0, 2, 3)
    #     return _mask

    # def iou_coef(self, smooth=1e-6):
    #     assert (self.pred.shape == self.actual.shape)
    #
    #     # axes = (2, 3)
    #     pred_onehot = self.mask2onehot(self.pred)  # N x K x H x W
    #     actual_onehot = self.mask2onehot(self.actual)  # N x K x H x W
    #
    #     intersection = torch.sum(torch.abs(pred_onehot * actual_onehot), dim=1, keepdim=True)  # N x 1 x H x W
    #     union = torch.sum(pred_onehot, dim=axes) + torch.sum(actual_onehot,  dim=1, keepdim=True) - intersection
    #
    #     intersection = torch.logical_and(pred_onehot, actual_onehot)
    #
    #     iou = torch.mean((intersection + smooth) / (union + smooth), dim=0)
    #     return torch.mean(iou)

    def get_mask(self, target, cls):

        '''

        Args:
            target: GT labels
            cls: Class of interest

        Returns:
            Mask that will filter out samples within a batch that
            doesn't include any pixels belonging to class of interest.

        '''

        B = target.shape[0]
        target = target.view(B, -1)  # B x N

        filter = (target == cls).float()  # B x N
        mask = (torch.sum(filter, dim=1) > 0)  # B
        return mask

    def compute_coef(self, pred, target, cls, mode='dice', smooth=1e-6):
        '''

        Args:
            pred: Prediction of the Model
            target: GT labels
            cls: Class of interest
            mode: IoU or DICE
            smooth: Ensure numerical stability

        Returns:
            Results of different metrics

        '''

        #assert cls != 0  # Not interested in background accuracy

        coef = 0.
        B = pred.shape[0]
        mask = self.get_mask(target, cls).float().unsqueeze(1).cpu()  # [B x 1]

        if mask.sum() < 1:
            #print('mask_sum < 1')
            return 0

        elif mask.float().sum() != 0 and (target == cls).float().sum() > 15:
            target = target.view(B, -1).cpu() * mask  # [B x N] * [B x 1]  --> [B x N]
            pred = pred.view(B, -1).cpu() * mask  # [B x N] * [B x 1] --> [B x N]

            pred_inds = pred == cls  # [B x N]
            target_inds = target == cls  # [B x N]

            intersection = (pred_inds * target_inds).long().sum().cpu()  # [B x N]
            union = pred_inds.long().sum().cpu() + target_inds.long().sum().cpu()  # [B x N]

            # Even now I ensure that union will not be 0 by filtering out images which don't belong to class of interest.
            if mode == 'dice':
                coef = 2. * float(intersection) / float(union + smooth)
            elif mode == 'iou':
                coef = float(intersection) / float(union - intersection + smooth)

            acc = (coef * mask).sum() / mask.sum()
            return acc

        else:
            #print('no coef computed')
            return 0
    # def compute_coef(self, pred, target, cls, mode, smooth=1e-6):
    #     '''
    #
    #     Args:
    #         pred: Prediction of the Model
    #         target: GT labels
    #         cls: Class of interest
    #         mode: IoU or DICE
    #         smooth: Ensure numerical stability
    #
    #     Returns:
    #         Results of different metrics
    #
    #     '''
    #
    #     coef = 0.
    #     B = pred.shape[0]
    #     mask = self.get_mask(target, cls)  # B
    #
    #     if mask.float().sum() < 1:
    #         return 0
    #
    #     target = target.view(B, -1)[mask, :].view(-1)
    #     pred = pred.view(B, -1)[mask, :].view(-1)
    #
    #     pred_inds = pred == cls  # N
    #     target_inds = target == cls  # N
    #
    #     intersection = (pred_inds[target_inds]).long().sum().cpu()  # Cast to long to prevent overflows
    #     union = pred_inds.long().sum().cpu() + target_inds.long().sum().cpu()
    #
    #     # Even now I ensure that union will not be 0 by filtering out images which don't belong to class of interest.
    #     if mode == 'dice':
    #         coef = 2.*float(intersection) / float(union + smooth)
    #     elif mode == 'iou':
    #         coef = float(intersection) / float(union - intersection + smooth)
    #
    #     acc = coef / mask.float().sum()
    #     return acc


    # def dice_coef(self, smooth=1e-6):
    #     assert (self.pred.shape == self.actual.shape)
    #
    #     axes = (2, 3)
    #     pred_onehot = self.mask2onehot(self.pred)
    #     actual_onehot = self.mask2onehot(self.actual)
    #
    #     intersection = torch.sum(pred_onehot * actual_onehot, dim=axes)
    #     union = torch.sum(pred_onehot, dim=axes) + torch.sum(actual_onehot, dim=axes)
    #     dice = torch.mean((2. * intersection + smooth) / (union + smooth + 1e-5), dim=0)
    #     return torch.mean(dice)
