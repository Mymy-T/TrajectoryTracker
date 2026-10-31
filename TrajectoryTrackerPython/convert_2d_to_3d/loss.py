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

    def __init__(self, eot_imbalance_weight, loss_weights):

        super().__init__()


        if len(loss_weights) not in [3, 4]:
            raise IndexError("loss_weights must have either length 3 (no height loss) or length 4 (w/ height loss).")

        self.loss_weights = loss_weights

        self.eot_loss_fn = EOTLoss(eot_imbalance_weight)
        self.trajectory_loss_fn = TrajectoryLoss()
        self.below_ground_loss_fn = BelowGroundLoss()
        self.height_loss_fn = HeightLoss()

    
    def forward(self, coords, eps, intermediate_heights, target_coords, target_eps, lengths):
        """
        - eps & target_eps are (N, L, 1)
        - coords & target_coords are (N, L, 3)
        - intermediate_heights is (N, L, 1)
        - lengths is (N,)
        """

        eot_loss = self.eot_loss_fn(eps, target_eps, lengths)
        trajectory_loss = self.trajectory_loss_fn(coords, target_coords, lengths)
        below_ground_loss = self.below_ground_loss_fn(coords, lengths)

        target_heights = target_coords[:, :, 1].unsqueeze(-1)
        height_loss = self.height_loss_fn(intermediate_heights, target_heights, lengths)

        total_loss = (self.loss_weights[0] * eot_loss) + (self.loss_weights[1] * trajectory_loss) + (self.loss_weights[2] * below_ground_loss)
        loss_components = {"EOT Loss": eot_loss, "Trajectory Loss": trajectory_loss, "Below Ground Loss": below_ground_loss, "Height Loss (not used/included)": height_loss}

        if len(self.loss_weights) == 4:
            total_loss += self.loss_weights[3] * height_loss
            loss_components = {"EOT Loss": eot_loss, "Trajectory Loss": trajectory_loss, "Below Ground Loss": below_ground_loss, "Height Loss": height_loss}
        

        return total_loss, loss_components


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
        pos_weight = torch.tensor(self.imbalance_weight, device=pred.device, dtype=pred.dtype)
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight, reduction="none") # (N, L)
        mask = _build_length_mask(lengths, pred.shape[1]) # (N, L)
        return bce[mask].sum() / mask.sum().clamp(min=1)
    

class BounceTokenLoss(nn.Module):

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
        pos_weight = torch.tensor(self.imbalance_weight, device=pred.device, dtype=pred.dtype)
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight, reduction="none") # (N, L)
        mask = _build_length_mask(lengths, pred.shape[1]) # (N, L)
        return bce[mask].sum() / mask.sum().clamp(min=1), {} # Extra dict represents loss components, which is empty in this case
    


class HeightLoss(nn.Module): # A loss for the intermediate heights within the network

    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, lengths):
        """
        - pred & target are (N, L, 1)
        - lengths is (N,)
        """
        sq_diffs = F.mse_loss(pred, target, reduction="none").squeeze(-1) # (N, L)
        mask = _build_length_mask(lengths, sq_diffs.shape[1]) # (N, L)
        mse = sq_diffs[mask].sum() / mask.sum().clamp(min=1)
        return mse


class TrajectoryLoss(nn.Module):

    def __init__(self, return_vector=False):
        super().__init__()

        self.return_vector = return_vector


    def forward(self, pred, target, lengths):
        """
        - pred & target are (N, L, 3)
        - lengths is (N,)
        """
        sq_diffs = F.mse_loss(pred, target, reduction="none") # (N, L, 3)
        per_timestep_error = torch.sum(sq_diffs, dim=-1) # (N, L) 
        mask = _build_length_mask(lengths, per_timestep_error.shape[1]) # (N, L)

        if not self.return_vector:
            mse = per_timestep_error[mask].sum() / mask.sum().clamp(min=1) # Scalar
        else:
            sse_vector = torch.where(mask, per_timestep_error, 0.0).sum(dim=-1) # (N,); mask index flattens it so dim=-1 works wrong
            denom_vector = mask.sum(dim=-1).clamp(min=1)
            mse = sse_vector / denom_vector # (N,)
        
        return mse # Scalar for most purposes, only vector of length N if corresponding arg passed


class BelowGroundLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, preds, lengths):
        """
        - pred is (N, L, 3)
        """
        y_coords = preds[:, :, 1]
        y_coords_copy = y_coords.clone()
        valid = _build_length_mask(lengths, y_coords.shape[1])
        y_coords_copy[~valid] = 0
        y_coords_copy = torch.where(y_coords_copy >= 0, 0, y_coords_copy)**2
        num_negative_y = (y_coords_copy != 0).sum().item()

        if num_negative_y == 0:
            return torch.zeros((), device=preds.device, dtype=preds.dtype)

        error = y_coords_copy.sum() / num_negative_y

        return error
