import torch
import torch.nn as nn
import torch.nn.functional as F


class FullLoss(nn.Module):

    def __init__(self, eot_imbalance_weight, loss_weights=None):

        super().__init__()

        if loss_weights is None:
            loss_weights = [10, 1, 10]

        if len(loss_weights) != 3:
            raise IndexError("loss_weights must have length 3.")

        self.loss_weights = loss_weights

        self.eot_loss_fn = EOTLoss(eot_imbalance_weight)
        self.trajectory_loss_fn = TrajectoryLoss()
        self.below_ground_loss_fn = BelowGroundLoss()

    
    def forward(self, coords, eps, target_coords, target_eps):
        """
        - eps & target_eps are (N, L, 1) 
        - coords & target_coords are (N, L, 3)
        """

        eot_loss = self.eot_loss_fn(eps, target_eps)
        trajectory_loss = self.trajectory_loss_fn(coords, target_coords)
        below_ground_loss = self.below_ground_loss_fn(coords)

        total_loss = (self.loss_weights[0] * eot_loss) + (self.loss_weights[1] * trajectory_loss) + (self.loss_weights[2] * below_ground_loss)

        return total_loss, (eot_loss, trajectory_loss, below_ground_loss)



class EOTLoss(nn.Module):

    def __init__(self, imbalance_weight):
        super().__init__()
        self.imbalance_weight = imbalance_weight
    

    def forward(self, pred, target):
        """
        - pred is (N, L, 1)
        - target is (N, L, 1)
        """
        pred = pred.squeeze(-1) # (N, L)
        target = target.squeeze(-1) # (N, L)
        weight = torch.where(target == 1, self.imbalance_weight, 1.0) # Per-element weight up-weighting the positive (EOT) class
        bce = F.binary_cross_entropy(pred, target, weight=weight) # NOTE: pred_eps is already masked to 0 for all elements beyond length; ground truth will also have them masked as 0, producing 0 loss
        return bce


class TrajectoryLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, pred, target):
        """
        - pred & target are (N, L, 3)
        """
        sq_diffs = F.mse_loss(pred, target, reduction="none") # (N, L, 3)
        mse = torch.mean(torch.sum(sq_diffs, dim=-1))
        return mse


class BelowGroundLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred):
        """
        - pred is (N, L, 3)
        """
        y_coords = pred[:, :, 1]
        y_coords = torch.where(y_coords >= 0, 0, y_coords)**2
        num_negative_y = (y_coords != 0).sum().item()

        if num_negative_y == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)

        error = y_coords.sum() / num_negative_y

        return error
