import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_length_mask(lengths, max_len):
    """
    - lengths is (N,)
    Returns a (N, L) boolean tensor, True at valid (non-padded) timesteps
    """
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0) # (1, L)
    return positions < lengths.unsqueeze(-1) # (N, L) from broadcasting


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

    
    def forward(self, coords, eps, target_coords, target_eps, lengths):
        """
        - eps & target_eps are (N, L, 1)
        - coords & target_coords are (N, L, 3)
        - lengths is (N,)
        """

        eot_loss = self.eot_loss_fn(eps, target_eps, lengths)
        trajectory_loss = self.trajectory_loss_fn(coords, target_coords, lengths)
        below_ground_loss = self.below_ground_loss_fn(coords)

        total_loss = (self.loss_weights[0] * eot_loss) + (self.loss_weights[1] * trajectory_loss) + (self.loss_weights[2] * below_ground_loss)

        return total_loss, (eot_loss, trajectory_loss, below_ground_loss)



class EOTLoss(nn.Module):

    def __init__(self, imbalance_weight):
        super().__init__()
        self.imbalance_weight = imbalance_weight
    

    def forward(self, pred, target, lengths):
        """
        - pred is (N, L, 1)
        - target is (N, L, 1)
        - lengths is (N,)
        """
        pred = pred.squeeze(-1) # (N, L)
        target = target.squeeze(-1) # (N, L)
        weight = torch.where(target == 1, self.imbalance_weight, 1.0) # Per-element weight for positive (EOT) class
        bce = F.binary_cross_entropy(pred, target, weight=weight, reduction="none") # (N, L)
        mask = _build_length_mask(lengths, pred.shape[1]) # (N, L)
        return bce[mask].sum() / mask.sum().clamp(min=1)


class TrajectoryLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, pred, target, lengths):
        """
        - pred & target are (N, L, 3)
        - lengths is (N,)
        """
        sq_diffs = F.mse_loss(pred, target, reduction="none") # (N, L, 3)
        per_timestep_error = torch.sum(sq_diffs, dim=-1) # (N, L) 
        mask = _build_length_mask(lengths, per_timestep_error.shape[1]) # (N, L)
        mse = per_timestep_error[mask].sum() / mask.sum().clamp(min=1)
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
