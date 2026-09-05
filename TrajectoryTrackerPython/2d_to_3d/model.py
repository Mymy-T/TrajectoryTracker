from networkx import diameter
from data_models import BallProperties
import numpy as np
from projector import dehomogenize
import settings
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _safe_divide(numerator, denominator, eps=1e-8):
    sign = torch.where(denominator >= 0, 1.0, -1.0)
    safe_denom = torch.where(denominator.abs() < eps, sign * eps, denominator)
    return numerator / safe_denom


def _build_validity_mask(lengths, max_len, remove_last_step=False):
    """
    - lengths is (N,)
    Returns a (N, L) boolean tensor, False at all padded timesteps
    """
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # (1, L)
    lengths = lengths.unsqueeze(-1) # (N, 1)
    return positions < (lengths if not remove_last_step else lengths - 1) # (N, L) from broadcasting; < due to zero indexing


def _reverse_valid_prefix(x, lengths):
    """
    -  x is (N, L, D)
    - lengths is (N,)
    Returns reversed sequence, ignoring padding (which remains at end)
    """
    N, L, D = x.shape[0], x.shape[1], x.shape[2]

    positions = torch.arange(L, device=x.device).unsqueeze(0).expand(N, L)  # (N, L)
    lengths = lengths.to(x.device).unsqueeze(1)  # (N, 1)

    reversed_positions = lengths - positions - 1 # These are indices of which element should be moved to the corresponding position

    is_valid = positions < lengths
    indices = torch.where(is_valid, reversed_positions, positions).unsqueeze(-1)  # (N, L, 1); padding stays at end
    indices = indices.expand(-1, -1, D) # (N, L, D)

    return torch.gather(x, dim=1, index=indices) # (N, L, D); looks up the corresponding reversed/padding indicies for the L dimension


def _mask(x, mask):
    """
    Replaces padded positions with zero
    - x is (N, L, D)
    - mask is (N, L)
    """
    return torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))


def _run_lstm_with_packing(lstm, x, lengths):
    """
    Returns a padded (N, L, D*2) tensor (for bidirectional lstms); Pytorch natural pack padding functionality
    """
    packed_in = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    packed_out, _ = lstm(packed_in)
    out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
    return out


# MARK: Full Network
class TrajectoryReconstructionNetwork(nn.Module):

    def __init__(self): # intrinsics is 3 x 3; extrinsincs is 4 x 4 # TODO: Updated all calls to instantiation
        super().__init__()

        self.eot_network = EOTNetwork()
        self.height_network = HeightNetwork()
        self.refinement_network = RefinementNetwork()

        self.to(dtype=settings.torch_dtype)


    def forward(self, x, lengths, intrinsics: torch.Tensor, extrinsics: torch.Tensor): # x is (N, L, 2); sequence of image coords; lengths is (N,); intrinsics is (N, 3, 3); extrinsics is (N, 4, 4)
        lengths = lengths.to(x.device)
        L = x.shape[1]
        mask = _build_validity_mask(lengths, L) # (N, L)

        c, d = self.obtain_ray_representation(x, intrinsics, extrinsics)
        p = self.obtain_intersection_representation(c, d) # (N, L, 4)
        p = _mask(p, mask) # Zero out padded positions

        delta_p = torch.zeros_like(p)
        delta_p[:, :-1] = p[:, 1:] - p[:, :-1] # Differencing; (N, L, 4)
        difference_mask = _build_validity_mask(lengths, L, remove_last_step=True) # Zero every padded position and each sequence's own last real step (no P_t+1 for last step)
        delta_p = _mask(delta_p, difference_mask)

        eot_input = delta_p # (N, L, 4)
        epsilons = self.eot_network(eot_input, lengths) # (N, L, 1)
        heights = self.height_network(p, delta_p, epsilons, lengths) # (N, L, 1)

        coords = self.obtain_coords_from_height(c, d, heights) # Find intersection of ray with height to get 3d coords; (N, L, 3)

        refinement_input = torch.cat((coords, p), dim=-1) # (N, L, 7)
        coord_deltas = self.refinement_network(refinement_input, lengths) # (N, L, 3)

        final_coords = coords + coord_deltas
        final_coords = _mask(final_coords, mask)
        epsilons = _mask(epsilons, mask)

        return final_coords, epsilons # (N, L, 3), (N, L, 1)


    def obtain_ray_representation(self, x: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor):
        """
        x is (N, L, 2); sequence of image coords
        intrinsics is (N, 3, 3)
        extrinsics is (N, 4, 4)
        Returns c, d; both of shapes (N, L, 3)
        """
        N, L = x.shape[0], x.shape[1]
        E_inv = torch.linalg.inv(extrinsics) # Maps cam coords to world coords in homogenous form; (4, 4)

        origin_homogeneous = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=E_inv.dtype, device=E_inv.device)
        c_batch = dehomogenize(torch.matmul(E_inv, origin_homogeneous)) # (N, 3)
        c = c_batch.unsqueeze(1).expand(N, L, 3)  # (N, L, 3)

        fx = intrinsics[:, 0, 0].unsqueeze(-1)  # (N, 1)
        fy = intrinsics[:, 1, 1].unsqueeze(-1)
        cx = intrinsics[:, 0, 2].unsqueeze(-1)
        cy = intrinsics[:, 1, 2].unsqueeze(-1)

        d_X_cam = (x[:, :, 0] - cx) / fx  # (N, L)
        d_Y_cam = (x[:, :, 1] - cy) / fy  # (N, L)
        d_Z_cam = torch.ones((N, L), dtype=x.dtype, device=x.device)
        d_extra_cam = torch.zeros((N, L), dtype=x.dtype, device=x.device) # Homogeneous direction coord
        d_cam = torch.stack((d_X_cam, d_Y_cam, d_Z_cam, d_extra_cam), dim=-1) # (N, L, 4)
        d_cam = d_cam.unsqueeze(-1) # (N, L, 4, 1)

        # Squeeze because matmul final shape is (N, L, 4, 1); E_inv changes from (N, 1, 4, 4) to (N, L, 4, 4) by broadcasting
        d = dehomogenize(torch.matmul(E_inv.unsqueeze(1), d_cam).squeeze(-1)) # (N, L, 3)

        return c, d


    def obtain_intersection_representation(self, c: torch.Tensor, d: torch.Tensor):
        """
        c & d both (N, L, 3)
        """
        c_x, c_y, c_z = c[:, :, 0], c[:, :, 1], c[:, :, 2] # (N, L)
        d_x, d_y, d_z = d[:, :, 0], d[:, :, 1], d[:, :, 2] # (N, L)

        pg_x = c_x - c_y * _safe_divide(d_x, d_y) # (N, L)
        pg_z = c_z - c_y * _safe_divide(d_z, d_y) # (N, L)
        pv_y = c_y - c_x * _safe_divide(d_y, d_x) # (N, L)
        pv_z = c_z - c_x * _safe_divide(d_z, d_x) # (N, L)

        return torch.stack((pg_x, pg_z, pv_y, pv_z), dim=-1) # (N, L, 4)
    

    def estimate_height_from_diameters(self, c: torch.Tensor, d: torch.Tensor, diameters: torch.Tensor, intrinsics: torch.Tensor):
        """
        c & d are (N, L, 3)
        diameters is (N, 2); first and last for each trajectory
        intrinsics is (N, 3, 3)
        Returns h, which is (N, 2), and contains estimated initial and final heights for each trajectory
        """
        true_diameter = 2 * BallProperties().R

        fx = intrinsics[:, 0, 0].unsqueeze(-1)  # (N, 1)
        fy = intrinsics[:, 1, 1].unsqueeze(-1)
        f = (fx + fy) / 2

        s = (f * true_diameter) / diameters # (N, 2)

        c_first = c[:, 0, :] # All (N, 3)
        d_first = d[:, 0, :]
        c_last = c[:, -1, :]
        d_last = d[:, -1, :]

        s_first = s[:, 0].unsqueeze(-1) # Both (N, 1)
        s_last = s[:, -1].unsqueeze(-1)

        first_coords = c_first + d_first * s_first # Both (N, 3)
        last_coords = c_last + d_last * s_last

        h_first = first_coords[:, 1] # Both (N,)
        h_last = last_coords[:, 1]
        h = torch.stack((h_first, h_last), dim=-1) # (N, 2)

        return h # (N, 2)


    def obtain_coords_from_height(self, c: torch.Tensor, d: torch.Tensor, h: torch.Tensor):
        """
        c & d are (N, L, 3)
        h is (N, L, 1)
        """
        c_x, c_y, c_z = c[:, :, 0], c[:, :, 1], c[:, :, 2] # (N, L)
        d_x, d_y, d_z = d[:, :, 0], d[:, :, 1], d[:, :, 2] # (N, L)

        h = h.squeeze(-1) # (N, L)

        x = c_x + (h - c_y) * _safe_divide(d_x, d_y) # (N, L)
        z = c_z + (h - c_y) * _safe_divide(d_z, d_y) # (N, L)

        return torch.stack((x, h, z), dim=-1) # (N, L, 3)


# MARK: EOT Network
class EOTNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm_0 = nn.LSTM(input_size=4, hidden_size=64, bidirectional=True, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True, batch_first=True)

        fc_0 = nn.Linear(in_features=128, out_features=32)
        fc_1 = nn.Linear(in_features=32, out_features=32)
        fc_2 = nn.Linear(in_features=32, out_features=32)
        fc_3 = nn.Linear(in_features=32, out_features=1)

        self.fc_network = nn.Sequential(
            fc_0,
            nn.LeakyReLU(negative_slope=0.01),
            fc_1,
            nn.LeakyReLU(negative_slope=0.01),
            fc_2,
            nn.LeakyReLU(negative_slope=0.01),
            fc_3,
            nn.Sigmoid()
        )

    def forward(self, x, lengths): # x is (N, L, 4); lengths is (N,)
        lstm_out_0 = _run_lstm_with_packing(self.lstm_0, x, lengths) # (N, L, 128)
        lstm_out_1 = _run_lstm_with_packing(self.lstm_1, lstm_out_0, lengths) # (N, L, 128)
        lstm_out_2 = _run_lstm_with_packing(self.lstm_2, lstm_out_1, lengths) # (N, L, 128)

        epsilons = self.fc_network(lstm_out_0 + lstm_out_2) # (N, L, 1); add residual from lstm_0
        epsilons = _mask(epsilons, _build_validity_mask(lengths, x.shape[1]))

        return epsilons  # (N, L, 1)


# MARK: Height Delta LSTM
class HeightDeltaLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm_0 = nn.LSTM(input_size=6, hidden_size=64, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)

        fc_0 = nn.Linear(in_features=64, out_features=32)
        fc_1 = nn.Linear(in_features=32, out_features=32)
        fc_2 = nn.Linear(in_features=32, out_features=32)
        fc_3 = nn.Linear(in_features=32, out_features=1)

        self.fc_network = nn.Sequential(
            fc_0,
            nn.LeakyReLU(negative_slope=0.01),
            fc_1,
            nn.LeakyReLU(negative_slope=0.01),
            fc_2,
            nn.LeakyReLU(negative_slope=0.01),
            fc_3
        )

    def forward(self, x_reduced): # x_reduced has shape (N, L, 5)
        N, L, _ = x_reduced.shape

        h = torch.zeros((N, 1), dtype=x_reduced.dtype, device=x_reduced.device)

        heights = []

        state_0 = None
        state_1 = None
        state_2 = None

        for t in range(L):
            x_t = torch.concatenate((x_reduced[:, t, :], h), dim=-1) # (N, 6)
            x_t = x_t.unsqueeze(dim=1) # (N, 1, 6); add sequence dimension

            x_t, state_0 = self.lstm_0(x_t, state_0)
            x_t, state_1 = self.lstm_1(x_t, state_1)
            x_t, state_2 = self.lstm_2(x_t, state_2) # x_t is now (N, 1, 64)

            delta_h_t = self.fc_network(x_t) # (N, 1, 1)
            delta_h_t = delta_h_t.squeeze(1) # (N, 1)

            heights.append(h) # Assumes the initial height is 0
            h = h + delta_h_t # Avoid += to avoid mutating the tensor already referenced in heights

        heights_tensor = torch.stack(heights, dim=1) # (N, L, 1)

        return heights_tensor


# MARK: Height LSTM
class HeightLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm_0 = nn.LSTM(input_size=5, hidden_size=64, bidirectional=True, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True, batch_first=True)

        fc_0 = nn.Linear(in_features=128, out_features=32)
        fc_1 = nn.Linear(in_features=32, out_features=32)
        fc_2 = nn.Linear(in_features=32, out_features=32)
        fc_3 = nn.Linear(in_features=32, out_features=1)

        self.fc_network = nn.Sequential(
            fc_0,
            nn.LeakyReLU(negative_slope=0.01),
            fc_1,
            nn.LeakyReLU(negative_slope=0.01),
            fc_2,
            nn.LeakyReLU(negative_slope=0.01),
            fc_3
        )

    def forward(self, x, lengths): # x has shape (N, L, 5); lengths is (N,)
        lstm_out_0 = _run_lstm_with_packing(self.lstm_0, x, lengths) # (N, L, 128)
        lstm_out_1 = _run_lstm_with_packing(self.lstm_1, lstm_out_0, lengths) # (N, L, 128)
        lstm_out_2 = _run_lstm_with_packing(self.lstm_2, lstm_out_1, lengths) # (N, L, 128)

        h_refined = self.fc_network(lstm_out_0 + lstm_out_2) # (N, L, 1); add residual from lstm_0
        h_refined = _mask(h_refined, _build_validity_mask(lengths, x.shape[1]))

        return h_refined  # (N, L, 1)


# MARK: Height Network
class HeightNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm_f = HeightDeltaLSTM()
        self.lstm_b = HeightDeltaLSTM()
        self.lstm_height = HeightLSTM()

    def forward(self, p, delta_p, epsilon, lengths): # p is (N, L, 4), delta_p is (N, L, 4), epsilon is (N, L, 1); lengths is (N,)
        N, L = p.shape[0], p.shape[1]

        lstm_f_input = torch.concatenate((delta_p, epsilon), dim=-1) # (N, L, 5); initial height feature added inside implmentation of HeightDeltaLSTM
        lstm_b_input = _reverse_valid_prefix(lstm_f_input, lengths)

        h_f = self.lstm_f(lstm_f_input) # (N, L, 1)
        h_b = self.lstm_b(lstm_b_input) # (N, L, 1); accumulated in reversed order
        h_b = _reverse_valid_prefix(h_b, lengths) # Reverse it again back to proper order

        positions = torch.arange(L, device=p.device, dtype=p.dtype).unsqueeze(0) # (1, L); zero-indexed
        denom = (lengths - 1).clamp(min=1).to(device=p.device, dtype=p.dtype).unsqueeze(1) # (N, 1); floored to avoid division by 0 for length 1 sequences
        fb_weights = (positions / denom).unsqueeze(-1) # (N, L, 1); reaches 1.0 at each sequence end, ignoring padding
        h = (1 - fb_weights) * h_f + fb_weights * h_b # (N, L, 1)

        lstm_height_input = torch.concatenate((p, h), dim=-1)
        h_refined = self.lstm_height(lstm_height_input, lengths) # (N, L, 1)

        return h_refined  # (N, L, 1)


# MARK: Refinement Network
class RefinementNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm_0 = nn.LSTM(input_size=7, hidden_size=64, bidirectional=True, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True, batch_first=True)

        fc_0 = nn.Linear(in_features=128, out_features=32)
        fc_1 = nn.Linear(in_features=32, out_features=32)
        fc_2 = nn.Linear(in_features=32, out_features=32)
        fc_3 = nn.Linear(in_features=32, out_features=3)

        self.fc_network = nn.Sequential(
            fc_0,
            nn.LeakyReLU(negative_slope=0.01),
            fc_1,
            nn.LeakyReLU(negative_slope=0.01),
            fc_2,
            nn.LeakyReLU(negative_slope=0.01),
            fc_3
        )

    def forward(self, x, lengths): # x is (N, L, 7); lengths is (N,)
        lstm_out_0 = _run_lstm_with_packing(self.lstm_0, x, lengths) # (N, L, 128)
        lstm_out_1 = _run_lstm_with_packing(self.lstm_1, lstm_out_0, lengths) # (N, L, 128)
        lstm_out_2 = _run_lstm_with_packing(self.lstm_2, lstm_out_1, lengths) # (N, L, 128)

        position_deltas = self.fc_network(lstm_out_0 + lstm_out_2) # (N, L, 3); add residual from lstm_0
        position_deltas = _mask(position_deltas, _build_validity_mask(lengths, x.shape[1]))

        return position_deltas  # (N, L, 3)
