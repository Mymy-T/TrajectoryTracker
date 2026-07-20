# from asyncio import WriteTransport
# import glob
# from tabnanny import check
from dataset import TrackNetDataset, DatasetType
from enum import Enum
import math
import os
from paths import ProjectPaths
# from statistics import NormalDist
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import train_settings

# TODO: Log hyperparam tuning results to a csv file for easy access, probably within paths.hyperparam_tuning_training_logs

device = train_settings.device

class RunType(Enum):
    final_model_training = 0
    hyperparameter_tuning = 1

# TODO: Write performance to csv file if run_type is hyperparameter tuning
def train_and_evaluate_model(model, lr, batch_size, checkpoints, max_epochs, name, run_type: RunType, verbose=True):

    # def _compute_wilson_interval(p, n, conf=0.95):
    #     alpha = 1 - conf
    #     pos_z_prob = conf + ((1 - conf) / 2) # prob from the left tail end, added to the "inner" area for conf interval; using inverse cdf gives the positive z bound, which is equal to |z|
    #     z = NormalDist().inv_cdf(pos_z_prob)

    #     denom = 1 + (z**2 / n)
    #     p_adjusted = p + (z**2 / (2*n))
    #     stdev_adjusted = z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))

    #     lower_bound = (p_adjusted - stdev_adjusted) / denom
    #     upper_bound = (p_adjusted + stdev_adjusted) / denom

    #     return lower_bound, upper_bound
        


    def _compare_model_performance(checkpoint_model: tuple, new_model: tuple, precision_diff_threshold=0.001, min_recall=0.8): # TODO: Get baseline min_recall by comparing it to pretrained model performance, if it's not too low
        """
        # OLD: checkpoint_model, new_model - tuples of (tp, tp+fp, mde, recall)
        checkpoint_model, new_model - tuples of (precision, mde, recall, epoch)
        returns True is new model is better, False otherwise
        """
        # Check recalls
        if new_model[2] < min_recall:
            if not checkpoint_model[2] < min_recall: # If the checkpoint model also has bad recall, continue and check precision, else return False
                return False
            
        # Check precision; wilson confidence interval overlap at 95% confidence used as rough comparison measure
        # p_checkpoint = checkpoint_model[0] / checkpoint_model[1]
        # n_checkpoint = checkpoint_model[1]
        # p_new = new_model[0] / new_model[1]
        # n_new = new_model[1]
        # lb_checkpoint, ub_checkpoint = _compute_wilson_interval(p_checkpoint, n_checkpoint)
        # lb_new, ub_new = _compute_wilson_interval(p_new, n_new)
        # overlapping = not ((ub_checkpoint < lb_new) or (ub_new < lb_checkpoint))
        if new_model[0] > checkpoint_model[0] + precision_diff_threshold:
            return True
        elif abs(checkpoint_model[0] - new_model[0]) <= precision_diff_threshold:
            return new_model[1] < checkpoint_model[1] # Check mde
        
        return False


    paths = ProjectPaths()
    if run_type == RunType.final_model_training:
        log_path = os.path.join(paths.final_model_training_logs, name)
        model_save_folder = os.path.join(paths.final_model_save_folder, name)
    else:
        log_path = os.path.join(paths.hyperparam_tuning_training_logs, name)
        model_save_folder = os.path.join(paths.hyperparam_tuning_model_save_folder, name)

    if not os.path.exists(log_path):
        os.mkdir(log_path)
        writer = SummaryWriter(log_dir=log_path)
    else:
        print(f"ERROR: A run with the same log name ({name}) has already been saved. Please either delete the previous run, or rename this run.")
        raise

    if not os.path.exists(model_save_folder):
        # base path is final_models/name; inside, there is a "checkpoints" folder to save all model checkpoints, and a model save file for the best model from the "checkpoints" folder
        os.mkdir(model_save_folder)
        checkpoints_folder = os.path.join(model_save_folder, "checkpoints")
        os.mkdir(checkpoints_folder)
    else:
        print(f"ERROR: A model with the same name ({name}) has already been saved. Please either delete the models from the previous run, or rename this run.")
        raise

    train_data = TrackNetDataset(DatasetType.train)
    valid_data = TrackNetDataset(DatasetType.valid)
    test_data = TrackNetDataset(DatasetType.test)

    running_on_cpu = (device.type == "cpu") # Pin memory if a gpu is present; this prevents allocated tensors from being moved to disk, speeding up how fast the tensors can be forwarded to the gpu
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=(not running_on_cpu))
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=(not running_on_cpu))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=(not running_on_cpu))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_model_metrics = [] # list of tuples of (tp, tp+fp, mean_dist_error, recall); tp, tp+fp used to calculate and compare precision, which is most important; then mde, then recall as a sanity check (make sure model isn't just skipping everything to have high precision)
    for epoch in range(1, max_epochs + 1):
        train_loss = _train_epoch(model, optimizer, loss_fn, train_dataloader, epoch, verbose=verbose)
        valid_loss, tp, tn, fp, fn, te = _evaluate(model, loss_fn, valid_dataloader, valid=True, verbose=verbose)
        precision, mde, recall = compute_eval_metrics(tp, tn, fp, fn, te)
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"Train": train_loss, "Valid": valid_loss}, global_step=epoch) # Graph shows loss for both train & valid set (used to check for overfitting/underfitting)
        writer.add_scalars(main_tag="Eval Metrics", tag_scalar_dict={"Precision": precision, "MDE": mde, "Recall": recall}, global_step=epoch) # Graph for validation set only
        writer.flush()

        if epoch <= checkpoints:
            best_model_metrics.append((precision, mde, recall, epoch))
            torch.save(model.state_dict(), os.path.join(checkpoints_folder, f"epoch_{epoch}"))
        else:
            worst_idx = 0
            for i in range(1, len(best_model_metrics)):
                if _compare_model_performance(checkpoint_model=best_model_metrics[i], new_model=best_model_metrics[worst_idx]): # If true, worst_idx doesn't refer to worst model checkpoint
                    worst_idx = i

            new_model_better = _compare_model_performance(checkpoint_model=best_model_metrics[worst_idx], new_model=(precision, mde, recall))
            if new_model_better:
                # Delete worst checkpoint
                deleted_model_metrics = best_model_metrics.pop(worst_idx)
                model_to_delete = os.path.join(checkpoints_folder, f"epoch_{deleted_model_metrics[3]}")
                os.remove(model_to_delete)
                # Insert new checkpoint
                best_model_metrics.append((precision, mde, recall, epoch))
                torch.save(model.state_dict(), os.path.join(checkpoints_folder, f"epoch_{epoch}"))
    
    best_idx = 0
    for i in range(1, len(best_model_metrics)):
        if _compare_model_performance(checkpoint_model=best_model_metrics[worst_idx], new_model=best_model_metrics[i]): # If true, worst_idx doesn't refer to worst model checkpoint
            best_idx = i
    best_model_path = os.path.join(checkpoints_folder, f"epoch_{best_model_metrics[best_idx][3]}")
    model_state = torch.load(best_model_path, weights_only=True, map_location=device)
    model.load_state_dict(model_state) # Load the best model
    torch.save(model.state_dict(), os.path.join(model_save_folder, name)) # Eg name.pt within final save models/name/
    if run_type == RunType.final_model_training:
        test_loss, tp, tn, fp, fn, te = _evaluate(model, loss_fn, test_dataloader, valid=False, verbose=verbose)
        precision, mde, recall = compute_eval_metrics(tp, tn, fp, fn, te)
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"Test": test_loss}, global_step=1)
        writer.add_scalars(main_tag="Eval Metrics", tag_scalar_dict={"TestPrecision": precision, "TestMDE": mde, "TestRecall": recall}, global_step=1)
        writer.flush()
        writer.close()
        return test_loss, precision, mde, recall
    # else: # TODO: Hyperparm tuning (possibly)
    #     writer.close()
    #     return best_loss


def _obtain_batch_eval_stats(model, pred, label, center_diff_tolerance=5): # NOTE center_diff_tolerance is in 360 by 640 space, not 1080 by 1920 space; measured in pixels
    """
    In deployment, the model should ideally have a low prediction error for the center of the ball, when it is a true positive. It is fine if the model misses some balls, but not if it makes up ball data when there is none. Therefore, precision is used as the primary metric, with mean center error as the secondary metric. Recall is computed as a sanity check to prevent extremely low recall.
    returns tp, tn, fp, fn, mae
    tp - true positiive - ground truth has 1 ball, output of hough circles gives exactly 1 ball sufficiently localized
    tn - true negative - ground truth has 0 balls, output of hough circles gives either no balls or more than 1 ball (any output with >1 ball is considered a negative output)
    fp - false positive - ground truth has 0 balls, output of hough circles gives exactly 1 ball, regardless of location; also if ground truth has 1 ball, output of hough circles gives exactly 1 ball not sufficiently localized
    fn - false negative - ground truth has 1 ball, output of hough circles gives either no balls or more than 1 ball (any output with >1 ball is considered a negative output); also if ground truth has 1 ball, output of hough circles gives exactly 1 ball not sufficiently localized
    dist_error_sum - sum of distance error for the center coords; only calculated for tp examples
    """
    # pred has shape (N, c, h, w); label has shape (N, h, w)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    dist_error_sum = 0
    no_ball_heatmap = torch.zeros(label.shape[1:]).to(device)
    pred = model.process_output(pred) # Shape (N, h, w)
    pred = model.filter_by_threshold(pred)
    for i in range(label.shape[0]):
        gt_is_negative = torch.allclose(label[i, :], no_ball_heatmap, atol=1e-6)
        ball_point = model.obtain_ball_center_coords(pred[i])
        if ball_point is not None: # [x, y, radius]
            if gt_is_negative:
                fp += 1
            else:
                gt_flattened_max_idx = torch.argmax(label[i]) # Get brightest coord in heatmap; this is the always the gt center
                gt_x, gt_y = torch.unravel_index(gt_flattened_max_idx, label.shape[1:])
                x, y, _ = ball_point
                center_diff = math.sqrt((x - gt_x)**2 + (y - gt_y)**2)
                if center_diff > center_diff_tolerance:
                    fp += 1
                    fn += 1
                else:
                    tp += 1
                    dist_error_sum += center_diff
        else:
            if gt_is_negative:
                tn += 1
            else:
                fn += 1
    # mde = tot_error / tp if tp > 0 else None
    return tp, tn, fp, fn, dist_error_sum

def compute_eval_metrics(tp, tn, fp, fn, dist_error_sum):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    mde = dist_error_sum / tp if tp > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, mde, recall



def _train_epoch(model, optimizer, loss_fn, train_loader, epoch, verbose):
    model.train()
    total_loss = 0
    for idx, batch in enumerate(train_loader): # batch is (inputs, outputs)
        inputs = batch[0].to(device)
        outputs = batch[1].to(device)
        optimizer.zero_grad()
        pred = model(inputs)
        loss = loss_fn(pred, outputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if verbose:
            print(f"Avg loss for batch {idx + 1}: {loss.item():.4f}")
    avg_loss = total_loss/len(train_loader)
    if verbose:
        print(f"Avg loss for epoch {epoch}: {avg_loss:.4f}") # epoch is indexed from 1
        print("------------------------------------------------")
    return avg_loss
    

def _evaluate(model, loss_fn, data_loader, valid, verbose=True):
    model.eval()
    with torch.inference_mode():
        total_loss = 0
        tp_tot = 0
        tn_tot = 0
        fp_tot = 0
        fn_tot = 0 
        error_tot = 0 # sum of dists to gt center coords; only for those classified as tp
        for idx, batch in enumerate(data_loader): # batch is (inputs, outputs)
            inputs = batch[0].to(device)
            outputs = batch[1].to(device)
            pred = model(inputs)
            loss = loss_fn(pred, outputs)
            total_loss += loss.item()
            tp, tn, fp, fn, error = _obtain_batch_eval_stats(model, pred, outputs)
            tp_tot += tp
            tn_tot += tn
            fp_tot += fp
            fn_tot += fn
            error_tot += error
            if verbose:
                print(f"Avg loss for batch {idx + 1}: {loss.item():.4f}")
        avg_loss = total_loss/len(data_loader)
        precision, mde, recall = compute_eval_metrics(tp, tn, fp, fn, error_tot)
        if verbose:
            if valid:
                name = "validation"
            else:
                name = "test"
            print(f"Avg {name} loss: {avg_loss:.4f}")
            print(f"{name} precision: {precision:.4f}")
            print(f"Avg {name} dist error: {mde:.4f}")
            print(f"{name} recall: {recall:.4f}")
        
        return avg_loss, tp_tot, tn_tot, fp_tot, fn_tot, error_tot
