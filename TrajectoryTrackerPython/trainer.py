from dataset import TrackNetDataset, DatasetType
from enum import Enum
import math
import os
from paths import ProjectPaths
import time
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import train_settings

# TODO: Log hyperparam tuning results to a csv file for easy access, probably within paths.hyperparam_tuning_training_logs

device = train_settings.device

class RunType(Enum):
    final_model_training = 0
    hyperparameter_tuning = 1


class FocalLoss(nn.Module):
    """
    - Down-weights the loss contribution from easy, well-classified pixels (background class) and concentrates gradient on hard pixels
    - Counters pixel class imbalance; see original focal loss paper for gamma=2.0 default.
    """
    def __init__(self, gamma=2.0, weights=None):
        super().__init__()
        self.gamma = gamma
        self.weights = weights # Optional per-class (256,) tensor of static weights based on prior class distributions, composable with focal weighting

    def forward(self, pred, target): # Pred shape is (N, c, h, w), target is (N, h, w)
        ce = F.cross_entropy(pred, target, weight=self.weights, reduction="none") # (N, h, w); each element has l = -w_t * log(p_t)
        if self.weights is not None:
            w_t = self.weights[target] # w_t is the weight of the corresponding label index
            p_t = torch.exp(-ce / w_t) # p_t is prob of correct class
        else: 
            p_t = torch.exp(-ce)
        return ((1 - p_t) ** self.gamma * ce).mean()


def _freeze_batchnorm(model):
    # Keeps pretrained BatchNorm running stats fixed
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


# NOTE: Saves resume_checkpoint every batch_size batches, and on every last batch of epoch
def train_and_evaluate_model(model, lr, batch_size, checkpoints, max_epochs, name, resuming, run_type: RunType, verbose=True, eval_batch_size=None, accumulation_steps=1, max_train_seconds=None):
    
    def _compare_model_performance(checkpoint_model: tuple, new_model: tuple, precision_diff_threshold=0.01, min_recall=0.8): # TODO: Get baseline min_recall by comparing it to pretrained model performance, if it's not too low
        """
        checkpoint_model, new_model - tuples of (precision, mde, recall, epoch)
        returns True is new model is better, False otherwise
        """
        # Check recalls
        if new_model[2] < min_recall:
            if not checkpoint_model[2] < min_recall: # If the checkpoint model also has bad recall, continue and check precision, else return False
                return False
            
        # Check precision
        if new_model[0] > checkpoint_model[0] + precision_diff_threshold:
            return True
        elif abs(checkpoint_model[0] - new_model[0]) <= precision_diff_threshold:
            mde_checkpoint = checkpoint_model[1] if checkpoint_model[1] is not None else float("inf")
            mde_new = new_model[1] if new_model[1] is not None else float("inf")
            return mde_new < mde_checkpoint # Check mde
        
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
    else:
        if not resuming:
            raise ValueError(f"A run with the same log name ({name}) has already been saved. Please either delete the previous run, rename this run, or set <resuming> to true.")
        else:
            print(f"Resuming a previous run ({name})...")
    writer = SummaryWriter(log_dir=log_path)

    checkpoints_folder = os.path.join(model_save_folder, "checkpoints")
    if not os.path.exists(model_save_folder):
        # Base path is final_models/name; inside, there is a "checkpoints" folder to save all model checkpoints, and a model save file for the best model from the "checkpoints" folder
        os.mkdir(model_save_folder)
        os.mkdir(checkpoints_folder)
    else:
        if not resuming:
            raise ValueError(f"A model with the same name ({name}) has already been saved. Please either delete the previous run, rename this run, or set <resuming> to true.")
        else:
            print(f"Resuming a previous run ({name})...")
    
    resume_checkpoint_path = os.path.join(model_save_folder, "resume_checkpoint.pt") # Saved inside final_models/name


    train_data = TrackNetDataset(DatasetType.train)
    valid_data = TrackNetDataset(DatasetType.valid)
    test_data = TrackNetDataset(DatasetType.test)

    eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size # eval/test don't need gradients, so they can safely use a larger batch than training within the same memory budget

    data_generator = torch.Generator() # Custom generator allows for saving train dataloader state
    sampler = RandomSampler(train_data, generator=data_generator)

    running_on_cpu = (device.type == "cpu") # Pin memory if a gpu is present; this prevents allocated tensors from being moved to disk, speeding up how fast the tensors can be forwarded to the gpu
    train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, pin_memory=(not running_on_cpu)) # Use custom sampler
    valid_dataloader = DataLoader(valid_data, batch_size=eval_batch_size, pin_memory=(not running_on_cpu))
    test_dataloader = DataLoader(test_data, batch_size=eval_batch_size, pin_memory=(not running_on_cpu))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = FocalLoss(gamma=2.0)

    start_epoch = 1
    best_model_metrics = [] # List of tuples of (precision, mean_dist_error, recall); precision is most important; then mde, then recall as a sanity check (make sure model isn't just skipping everything to have high precision)

    # Only used in _train_epoch
    resume_batch_idx = 0
    total_batch_loss = 0

    if resuming:
        if os.path.exists(resume_checkpoint_path):
            resume_checkpoint = torch.load(resume_checkpoint_path, map_location=device)

            model.load_state_dict(resume_checkpoint["model_state_dict"])
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            best_model_metrics = resume_checkpoint["best_model_metrics"]
            start_epoch = resume_checkpoint["epoch"]

            resume_batch_idx = resume_checkpoint["resume_batch_idx"] # idx of batch training will start on
            total_batch_loss = resume_checkpoint["total_batch_loss"]
            print(f"Resuming training from epoch {start_epoch}, starting with batch {resume_batch_idx + 1}") # add one since batch idx is 0 indexed
        else:
            raise FileNotFoundError(f"There is no existing resume_checkpoint.pt file. Ensure it is named correctly and at the correct path ({resume_checkpoint_path}) or run train_script.py with the resuming arg set to false.")


    training_start_time = time.monotonic() # Timer for a single training session (not cumulative over the entire training process)

    for epoch in range(start_epoch, max_epochs + 1):
        data_generator.manual_seed(0 + epoch)

        train_loss, time_limit_reached = _train_epoch(model, optimizer, loss_fn, train_dataloader, epoch, resume_batch_idx=resume_batch_idx, total_loss=total_batch_loss, resume_checkpoint_path=resume_checkpoint_path, verbose=verbose, accumulation_steps=accumulation_steps, best_model_metrics=best_model_metrics, max_train_seconds=max_train_seconds, training_start_time=training_start_time)

        # Reset back to initial state, resuming has been completed; everything should run normally again
        resume_batch_idx = 0
        total_batch_loss = 0

        if time_limit_reached:
            # Training is paused, not complete: skip validation, model selection, and finalization for this partial epoch; resume later using resume_checkpoint.pt
            writer.flush()
            writer.close()
            return None

        valid_loss, tp, tn, fp, fn, te = evaluate(model, loss_fn, valid_dataloader, valid=True, verbose=verbose)
        precision, mde, recall = compute_eval_metrics(tp, tn, fp, fn, te)

        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"Train": train_loss, "Valid": valid_loss}, global_step=epoch) # Graph shows loss for both train & valid set (used to check for overfitting/underfitting)
        eval_metrics = {"Precision": precision, "Recall": recall}
        if mde is not None:
            eval_metrics["MDE"] = mde
        writer.add_scalars(main_tag="Eval Metrics", tag_scalar_dict=eval_metrics, global_step=epoch) # Graph for validation set only; MDE omitted on steps with no true positives, rather than logging a spike
        writer.flush()

        # Selection of the saved best checkpoint models (resume checkpoint untouched)
        if epoch <= checkpoints:
            best_model_metrics.append((precision, mde, recall, epoch))
            torch.save(model.state_dict(), os.path.join(checkpoints_folder, f"epoch_{epoch}.pt"))
        else:
            worst_idx = 0
            for i in range(1, len(best_model_metrics)):
                if _compare_model_performance(checkpoint_model=best_model_metrics[i], new_model=best_model_metrics[worst_idx]): # If true, worst_idx doesn't refer to worst model checkpoint
                    worst_idx = i

            new_model_better = _compare_model_performance(checkpoint_model=best_model_metrics[worst_idx], new_model=(precision, mde, recall))
            if new_model_better:
                # Delete worst checkpoint
                deleted_model_metrics = best_model_metrics.pop(worst_idx)
                model_to_delete = os.path.join(checkpoints_folder, f"epoch_{deleted_model_metrics[3]}.pt")
                os.remove(model_to_delete)
                # Insert new checkpoint
                best_model_metrics.append((precision, mde, recall, epoch))
                torch.save(model.state_dict(), os.path.join(checkpoints_folder, f"epoch_{epoch}.pt"))
    
    best_idx = 0
    for i in range(1, len(best_model_metrics)):
        if _compare_model_performance(checkpoint_model=best_model_metrics[best_idx], new_model=best_model_metrics[i]): # If true, best_idx doesn't refer to best model checkpoint
            best_idx = i
    best_model_path = os.path.join(checkpoints_folder, f"epoch_{best_model_metrics[best_idx][3]}.pt")

    # Load the best model and save it for easy access
    model_state = torch.load(best_model_path, weights_only=True, map_location=device)
    model.load_state_dict(model_state)
    model_save_name = name + ".pt"
    torch.save(model.state_dict(), os.path.join(model_save_folder, model_save_name)) # Eg name.pt within final save models/name/

    if run_type == RunType.final_model_training:
        test_loss, tp, tn, fp, fn, te = evaluate(model, loss_fn, test_dataloader, valid=False, verbose=verbose)
        precision, mde, recall = compute_eval_metrics(tp, tn, fp, fn, te)

        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"Test": test_loss}, global_step=1)
        test_eval_metrics = {"TestPrecision": precision, "TestRecall": recall}
        if mde is not None:
            test_eval_metrics["TestMDE"] = mde
        writer.add_scalars(main_tag="Eval Metrics", tag_scalar_dict=test_eval_metrics, global_step=1)
        writer.flush()
        writer.close()

        os.remove(resume_checkpoint_path) # Remove resume_checkpoint.pt, everything has been completed for this run
        return test_loss, precision, mde, recall
    # else: # TODO: Hyperparm tuning (possibly)
    #     writer.close()
    #     return best_loss


def _obtain_batch_eval_stats(model, pred, label, center_diff_tolerance=5): # NOTE center_diff_tolerance is in 360 by 640 space, not 1080 by 1920 space; measured in pixels; 5 pixels also used in tracknet paper
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
    no_ball_heatmap = torch.zeros(label.shape[1:], dtype=label.dtype).to(device)
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
                gt_row, gt_col = torch.unravel_index(gt_flattened_max_idx, label.shape[1:]) # unravel_index on a (h, w) shape returns (row, col)
                x, y, _ = ball_point # HoughCircles returns (x=col/width, y=row/height, radius)
                center_diff = math.sqrt((x - gt_col)**2 + (y - gt_row)**2)
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
    return tp, tn, fp, fn, dist_error_sum



def compute_eval_metrics(tp, tn, fp, fn, dist_error_sum):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    mde = dist_error_sum / tp if tp > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, mde, recall



def _train_epoch(model, optimizer, loss_fn, train_loader, epoch, resume_batch_idx, total_loss, resume_checkpoint_path, verbose, accumulation_steps, best_model_metrics, max_train_seconds, training_start_time): # best_model_metrics passed in for only purpose of saving to resume_checkpoint
    model.train()
    _freeze_batchnorm(model) # Must be applied after every call to model.train()
    num_batches = len(train_loader)
    optimizer.zero_grad()
    time_limit_reached = False
    for idx, batch in enumerate(train_loader): # Batch is (inputs, outputs)
        if idx < resume_batch_idx:
            continue
        inputs = batch[0].to(device)
        outputs = batch[1].to(device)
        pred = model(inputs)
        loss = loss_fn(pred, outputs)
        loss = loss / accumulation_steps # Divide so accumulated gradients average over the effective batch, not sum
        loss.backward() # Accumulates gradients if zero grad isn't called
        is_last_batch = (idx == num_batches - 1)
        if (idx + 1) % accumulation_steps == 0 or is_last_batch: # Flush on the last batch too, so a trailing partial group still gets applied
            optimizer.step()
            optimizer.zero_grad()
            resume_checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "resume_batch_idx": idx + 1, # idx of the next batch, not the current completed batch
                "total_batch_loss": total_loss, # For logging purposes
                "best_model_metrics": best_model_metrics
            }
            torch.save(resume_checkpoint, resume_checkpoint_path)
            if max_train_seconds is not None and (time.monotonic() - training_start_time) >= max_train_seconds:
                # Stop right at this resume checkpoint save boundary
                print(f"Time budget of {max_train_seconds}s reached after finishing batch {idx + 1} of epoch {epoch}; pausing training. Resume later with resuming=True.")
                time_limit_reached = True
                
        total_loss += loss.item() * accumulation_steps # True, unscaled loss for logging

        if verbose:
            print(f"Epoch {epoch}; avg loss for batch {idx + 1}: {loss.item():.8f}")
        
        if time_limit_reached:
            break
    
    avg_loss = total_loss/len(train_loader)

    if not time_limit_reached:
        if verbose:
            print(f"Avg loss for epoch {epoch}: {avg_loss:.8f}") # Epoch is indexed from 1
            print("------------------------------------------------")

    return avg_loss, time_limit_reached # avg_loss unused if time_limit_reached
    

def evaluate(model, loss_fn, data_loader, valid, verbose=True):
    model.eval()
    print("Starting Evaluation...")
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
                print(f"Avg loss for batch {idx + 1}: {loss.item():.8f}")
        avg_loss = total_loss/len(data_loader)
        precision, mde, recall = compute_eval_metrics(tp_tot, tn_tot, fp_tot, fn_tot, error_tot)
        mde_str = f"{mde:.8f}" if mde is not None else "N/A"
        if verbose:
            if valid:
                name = "validation"
            else:
                name = "test"
            print(f"Avg {name} loss: {avg_loss:.8f}")
            print(f"TP: {tp_tot} \t TN: {tn_tot} \t FP: {fp_tot} \t FN: {fn_tot}")
            print(f"{name} precision: {precision:.8f}")
            print(f"Avg {name} dist error: {mde_str}")
            print(f"{name} recall: {recall:.8f}")
        
        return avg_loss, tp_tot, tn_tot, fp_tot, fn_tot, error_tot
