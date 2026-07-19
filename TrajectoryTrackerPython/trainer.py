from dataset import TrackNetDataset, DatasetType
from enum import Enum
import math
import os
from paths import ProjectPaths
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
def train_and_evaluate_model(model, lr, batch_size, min_epochs, max_epochs, early_stopping_patience, name, run_type: RunType, verbose=True):
    paths = ProjectPaths()
    if run_type == RunType.final_model_training:
        log_path = os.path.join(paths.final_model_training_logs, name)
        model_save_path = os.path.join(paths.final_model_save_folder, name)
    else:
        log_path = os.path.join(paths.hyperparam_tuning_training_logs, name)
        model_save_path = os.path.join(paths.hyperparam_tuning_model_save_folder, name)

    if not os.path.exists(log_path):
        os.mkdir(log_path)
        writer = SummaryWriter(log_dir=log_path)
    else:
        print(f"ERROR: A run with the same log name ({name}) has already been saved. Please either delete the previous run, or rename this run.")
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

    best_loss = float("inf")
    counter = 0 # Counts number of times valid_loss didn't improve from last epoch
    for epoch in range(1, max_epochs + 1):
        train_loss = _train_epoch(model, optimizer, loss_fn, train_dataloader, epoch, verbose=verbose)
        valid_loss = _evaluate(model, loss_fn, valid_dataloader, valid=True, verbose=verbose)
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"Train": train_loss, "Valid": valid_loss}, global_step=epoch)
        writer.flush()

        if best_loss > valid_loss:
            best_loss = valid_loss
            counter = 0
            torch.save(model.state_dict(), model_save_path) # Saves the best version; all other later saves override the previous model save
        else:
            if epoch <= min_epochs:
                continue
            counter += 1
        
        if counter >= early_stopping_patience:
            if verbose:
                print(f"Training of {name} stopped early on epoch {epoch} of {max_epochs}")
            break
    model_state = torch.load(model_save_path, weights_only=True, map_location=device)
    model.load_state_dict(model_state) # Load the best model
    if run_type == RunType.final_model_training:
        test_loss = _evaluate(model, loss_fn, test_dataloader, valid=False, verbose=verbose)
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"Test": test_loss}, global_step=1)
        writer.flush()
        writer.close()
        return test_loss
    else:
        writer.close()
        return best_loss


def _obtain_batch_eval_metrics(model, pred, label, center_diff_tolerance=5): # NOTE center_diff_tolerance is in 360 by 640 space, not 1080 by 1920 space
    """
    In deployment, the model should ideally have a low prediction error for the center of the ball, when it is a true positive. It is fine if the model misses some balls, but not if it makes up ball data when there is none. Therefore, precision is used as the primary metric, with mean center error as the secondary metric. Recall is computed as a sanity check to prevent extremely low recall.
    returns tp, tn, fp, fn, mae
    tp - true positiive - ground truth has 1 ball, output of hough circles gives exactly 1 ball sufficiently localized
    tn - true negative - ground truth has 0 balls, output of hough circles gives either no balls or more than 1 ball (any output with >1 ball is considered a negative output)
    fp - false positive - ground truth has 0 balls, output of hough circles gives exactly 1 ball, regardless of location; also if ground truth has 1 ball, output of hough circles gives exactly 1 ball not sufficiently localized
    fn - false negative - ground truth has 1 ball, output of hough circles gives either no balls or more than 1 ball (any output with >1 ball is considered a negative output); also if ground truth has 1 ball, output of hough circles gives exactly 1 ball not sufficiently localized
    mde - mean distance error for the center coords; only calculated for tp examples
    """
    # pred has shape (N, c, h, w); label has shape (N, h, w)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    tot_error = 0
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
                    tot_error += center_diff
        else:
            if gt_is_negative:
                tn += 1
            else:
                fn += 1
    mde = tot_error / tp if tp > 0 else None
    return tp, tn, fp, fn, mde

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
        for idx, batch in enumerate(data_loader): # batch is (inputs, outputs)
            inputs = batch[0].to(device)
            outputs = batch[1].to(device)
            pred = model(inputs)
            loss = loss_fn(pred, outputs)
            total_loss += loss.item()
            
            if verbose:
                print(f"Avg loss for batch {idx + 1}: {loss.item():.4f}")
        avg_loss = total_loss/len(data_loader)
        if verbose:
            if valid:
                name = "validation"
            else:
                name = "test"
            print(f"Avg {name} loss: {avg_loss:.4f}")
        
        return avg_loss
