import torch.nn.functional as F
import torch
from torch import nn
import inspect


class DiceLoss(nn.Module):
    def __init__(
        self,
        eps=1e-7,
        round_probs=False,
        background_weight=1.0,
        return_per_class_scores=False,
        per_image=False,
        logits=True,
    ):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.round_probs = round_probs
        self.background_weight = background_weight
        self.return_per_class_scores = return_per_class_scores
        self.round_probs = round_probs
        self.per_image = per_image
        self.logits = logits

    def forward(self, output, target):
        """
        Source: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

        Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            target: a tensor of shape [B, H, W].
            output: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
            round_probs: bool indicating  whether output class probabilities are rounded to 1 or not, when computing the dice score (by default,
            dice score is computed by multiplying GT masks with fractional probabilities)
            logits: Wether this expect logits as input or softmaxed logits (defaults to True, which means logits without any softmax)


        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        target_ = target.unsqueeze(1) if len(target.shape) == 3 else target
        num_classes = output.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[target_.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(output)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)

        else:
            true_1_hot = torch.eye(num_classes)[target_.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            if self.logits:
                probas = F.softmax(output, dim=1)
            else:
                probas = output

            if self.round_probs:
                probas = (
                    torch.eye(num_classes)[probas.argmax(dim=1).squeeze(1)]
                    .permute(0, 3, 1, 2)
                    .float()
                    .to(output.device)
                )

        true_1_hot = true_1_hot.type(output.type())

        dims = (
            tuple(range(2, target_.ndimension()))
            if self.per_image
            else (0,) + tuple(range(2, target_.ndimension()))
        )

        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)

        # Return per-class dis scores and a mask indicating which classes are present in the GT
        dice_score = 2.0 * intersection / (cardinality + self.eps)
        if self.return_per_class_scores:
            return dice_score, true_1_hot.sum(dims) > 0

        # We divide the score by the number of classes present (otherwise, the score is 0 whenever some class is not present)
        num_present_classes = (true_1_hot.sum(dims) > 0).float()
        num_present_classes = (
            num_present_classes if self.per_image else num_present_classes.sum()
        )
        if self.per_image:
            num_present_classes[:, 0, ...] = self.background_weight
        else:
            num_present_classes += self.background_weight - 1
        if self.per_image:
            dice_score[:, 0, ...] = dice_score[:, 0, ...] * self.background_weight
            dice_score = (dice_score.sum(dim=1)/(num_present_classes.sum(dim=1) + self.eps)).unsqueeze(1)

        else:
            # Note: this calculation consider the batch as a big image
            # Which means, we do not pay attention to per image dice,
            # we are rather interested in the average.
            dice_score[0] *= self.background_weight
            dice_score = dice_score.sum() / (num_present_classes + self.eps)
        return 1 - dice_score


class CrossEntropyLoss(nn.Module):
    def __init__(self, weighted=False, weights=None):
        super(CrossEntropyLoss, self).__init__()

        self.weighted = weighted
        self.weights = weights

    def forward(self, output, target):
        if not self.weighted:
            return F.cross_entropy(output, target)

        # Use given per-class weights
        elif self.weights:
            return F.cross_entropy(output, target, weight=self.weights)

        # Compute per-class weights based on the current target statistcs
        else:
            # Get the appearance frequency per class
            num_classes = output.shape[1]
            ce_weights = torch.zeros(num_classes).to(output.device)
            classes, class_count = torch.unique(target, return_counts=True)

            # Assign each class a weight inversely proportional to its frequency
            ce_weights[classes] = 1 / class_count.float()
            num_present_classes = (ce_weights != 0).sum()

            # Scale weights so that they add up to the number of present classes
            ce_weights = (num_present_classes / ce_weights.sum()) * ce_weights

            return F.cross_entropy(output, target, weight=ce_weights)


class CombinedLoss(nn.Module):
    """
    Combines an arbitrary sum of the losses above
    """

    def __init__(self, losses):
        """
        Instantiates the losses above
        :param losses: list of losses class names
        """
        super(CombinedLoss, self).__init__()
        self.losses = losses

    def forward(self, output, target):
        loss_sum = 0
        for loss in self.losses:
            loss_sum += loss(output, target)

        return loss_sum
