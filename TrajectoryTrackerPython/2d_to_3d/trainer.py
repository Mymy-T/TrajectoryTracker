from checkpointing import CheckpointManager, ResumeCheckpointManager, LossLog
from dataset import TrajectoryReconstructionDataset, DatasetType, homogenize_seq_length
from loss import FullLoss
import os
from paths import Paths
import time
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import settings

device = settings.device


def train_and_evaluate(model, lr, batch_size, max_checkpoints, max_epochs, name, resuming, verbose=True, max_train_seconds=None, train_dataset=None, valid_dataset=None, test_dataset=None):
    # MARK: Training Setup
    paths = Paths()

    log_path = os.path.join(paths.logs_folder, name)
    model_save_folder = os.path.join(paths.model_saves_folder, name)

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
        # Base path is model_saves/name/; inside, there is a "checkpoints" folder to save all model checkpoints, and a model save file for the best model from the "checkpoints" folder
        os.mkdir(model_save_folder)
        os.mkdir(checkpoints_folder)
    else:
        if not resuming:
            raise ValueError(f"A model with the same name ({name}) has already been saved. Please either delete the previous run, rename this run, or set <resuming> to true.")
        else:
            print(f"Resuming a previous run ({name})...")
    
    resume_checkpoint_path = os.path.join(model_save_folder, "resume_checkpoint.pt") # Saved inside models/name/
    resume_manager = ResumeCheckpointManager(resume_checkpoint_path)

    # Datasets
    train_data = train_dataset if train_dataset is not None else TrajectoryReconstructionDataset(DatasetType.train, noise_2d=True)
    valid_data = valid_dataset if valid_dataset is not None else TrajectoryReconstructionDataset(DatasetType.valid, noise_2d=True)
    test_data = test_dataset if test_dataset is not None else TrajectoryReconstructionDataset(DatasetType.test, noise_2d=True)

    data_generator = torch.Generator() # Custom generator allows for saving train dataloader state
    sampler = RandomSampler(train_data, generator=data_generator)

    running_on_cpu = (device.type == "cpu") # Pin memory if a gpu is present; this prevents allocated tensors from being moved to disk, speeding up how fast the tensors can be forwarded to the gpu
    train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, pin_memory=(not running_on_cpu), collate_fn=homogenize_seq_length) # Use custom sampler
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, pin_memory=(not running_on_cpu), collate_fn=homogenize_seq_length)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, pin_memory=(not running_on_cpu), collate_fn=homogenize_seq_length)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = FullLoss(eot_imbalance_weight=1.0) # TODO: Find eot imbalance weight

    # MARK: Training
    start_epoch = 1
    checkpoint_manager = CheckpointManager(checkpoints_folder, max_checkpoints) # Tracks/saves best_model_metrics (loss, epoch)

    # Only used in _train_epoch
    resume_batch_idx = 0
    loss_log = LossLog(0, (0, 0, 0))

    if resuming:
        resume_state = resume_manager.load(model, optimizer)
        checkpoint_manager.best_model_metrics = resume_state["best_model_metrics"]
        start_epoch = resume_state["start_epoch"]
        resume_batch_idx = resume_state["resume_batch_idx"]
        loss_log = resume_state["loss_info"]
    
    training_start_time = time.monotonic()

    for epoch in range(start_epoch, max_epochs + 1):
        data_generator.manual_seed(0 + epoch)

        best_model_metrics = checkpoint_manager.best_model_metrics if checkpoint_manager is not None else None
        train_loss, train_loss_components, time_limit_reached = _train_epoch(model, optimizer, loss_fn, train_dataloader, epoch, resume_batch_idx, loss_log, resume_manager, verbose, best_model_metrics, max_train_seconds, training_start_time)

        # Reset back to initial state, resuming has been completed; everything should run normally again
        resume_batch_idx = 0
        loss_log.total_loss = 0
        loss_log.eot_loss = 0
        loss_log.trajectory_loss = 0
        loss_log.below_ground_loss = 0

        if time_limit_reached:
            # Training is paused, not complete: skip validation, model selection, and finalization for this partial epoch; resume later using resume_checkpoint.pt
            writer.flush()
            writer.close()
            return None

        valid_loss, valid_loss_components = evaluate(model, loss_fn, valid_dataloader, valid=True, verbose=verbose)

        # Selection of the saved best checkpoint models
        checkpoint_manager.update(model, epoch, valid_loss)

        # Logging
        writer.add_scalars(main_tag="Total Loss", tag_scalar_dict={"Train": train_loss, "Valid": valid_loss}, global_step=epoch)
        writer.add_scalars(main_tag="EOT Loss", tag_scalar_dict={"Train": train_loss_components[0], "Valid": valid_loss_components[0]}, global_step=epoch)
        writer.add_scalars(main_tag="Trajectory Loss", tag_scalar_dict={"Train": train_loss_components[1], "Valid": valid_loss_components[1]}, global_step=epoch)
        writer.add_scalars(main_tag="Below Ground Loss", tag_scalar_dict={"Train": train_loss_components[2], "Valid": valid_loss_components[2]}, global_step=epoch)

        writer.flush()

    # MARK: Post Training

    best_model_path = checkpoint_manager.best_checkpoint_path()

    # Load the best model and save it for easy access
    model_state = torch.load(best_model_path, weights_only=True, map_location=device)
    model.load_state_dict(model_state)
    model_save_name = name + ".pt"
    torch.save(model.state_dict(), os.path.join(model_save_folder, model_save_name)) # name.pt within final save models/name/

    test_loss, test_loss_components = evaluate(model, loss_fn, test_dataloader, valid=False, verbose=verbose)

    writer.add_scalars(main_tag="Total Loss", tag_scalar_dict={"Test": test_loss}, global_step=1) 
    writer.add_scalars(main_tag="EOT Loss", tag_scalar_dict={"Test": test_loss_components[0]}, global_step=1)
    writer.add_scalars(main_tag="Trajectory Loss", tag_scalar_dict={"Test": test_loss_components[1]}, global_step=1)
    writer.add_scalars(main_tag="Below Ground Loss", tag_scalar_dict={"Test": test_loss_components[2]}, global_step=1)

    writer.flush()
    writer.close()

    resume_manager.delete()

    return test_loss, test_loss_components
    

# MARK: Train Epoch
def _train_epoch(model, optimizer, loss_fn, train_loader, epoch, resume_batch_idx, loss_log, resume_manager, verbose, best_model_metrics, max_train_seconds, training_start_time): # best_model_metrics passed in for only purpose of saving to resume_checkpoint
    model.train()
    optimizer.zero_grad()
    num_batches = len(train_loader)
    time_limit_reached = False

    total_loss = loss_log.total_loss
    eot_loss = loss_log.eot_loss
    trajectory_loss = loss_log.trajectory_loss
    below_ground_loss = loss_log.below_ground_loss

    for idx, batch in enumerate(train_loader): # Batch is (inputs, coord_labels, eot_labels, lengths)

        if idx < resume_batch_idx:
            continue

        inputs = batch[0].to(device) # (N, L, 2)
        coord_labels = batch[1].to(device) # (N, L, 3)
        eot_labels = batch[2].to(device) # (N, L, 1)
        lengths = batch[3].to(device) # (N,)
        intrinsics = batch[4].to(device) # (N, 3, 3)
        extrinsics = batch[5].to(device) # (N, 4, 4)


        pred_coords, pred_eot = model(inputs, lengths, intrinsics, extrinsics)

        loss, loss_components = loss_fn(pred_coords, pred_eot, coord_labels, eot_labels, lengths)
        loss.backward() # Accumulates gradients if zero grad isn't called

        total_loss += loss.item() # Aggregates losses for logging
        eot_loss += loss_components[0].item()
        trajectory_loss += loss_components[1].item()
        below_ground_loss += loss_components[2].item()

        optimizer.step()
        optimizer.zero_grad()

        loss_log = LossLog(total_loss, (eot_loss, trajectory_loss, below_ground_loss))
        resume_manager.save(model, optimizer, epoch, idx + 1, loss_log, best_model_metrics) # Resume on next batch

        if max_train_seconds is not None and (time.monotonic() - training_start_time) >= max_train_seconds:

            print(f"Time budget of {max_train_seconds}s reached after finishing batch {idx + 1} of epoch {epoch}; pausing training. Resume later with resuming=True.")

            time_limit_reached = True
        

        if verbose:
            print(f"Epoch {epoch}; avg loss for batch {idx + 1}: {loss.item():.8f}")
        

        if time_limit_reached:
            break
    

    avg_loss = total_loss / num_batches
    avg_eot_loss = eot_loss / num_batches
    avg_trajectory_loss = trajectory_loss / num_batches
    avg_below_ground_loss = below_ground_loss / num_batches


    if not time_limit_reached:
        if verbose:
            print(f"Avg loss for epoch {epoch}: {avg_loss:.8f}") # Epoch is indexed from 1
            print("------------------------------------------------")

    return avg_loss, (avg_eot_loss, avg_trajectory_loss, avg_below_ground_loss), time_limit_reached # avg_loss unused if time_limit_reached
    

# MARK: Evaluate
def evaluate(model, loss_fn, data_loader, valid, verbose=True):

    model.eval()
    num_batches = len(data_loader)

    if verbose:
        print("Starting Evaluation...")

    with torch.inference_mode():

        total_loss = 0
        eot_loss = 0
        trajectory_loss = 0
        below_ground_loss = 0

        for idx, batch in enumerate(data_loader): # Batch is (inputs, coord_labels, eot_labels, lengths)

            inputs = batch[0].to(device) # (N, L, 2)
            coord_labels = batch[1].to(device) # (N, L, 3)
            eot_labels = batch[2].to(device) # (N, L, 1)
            lengths = batch[3].to(device) # (N,)
            intrinsics = batch[4].to(device) # (N, 3, 3)
            extrinsics = batch[5].to(device) # (N, 4, 4)

            pred_coords, pred_eot = model(inputs, lengths, intrinsics, extrinsics) # (N, L, 3), (N, L, 1)

            loss, loss_components = loss_fn(pred_coords, pred_eot, coord_labels, eot_labels, lengths)

            total_loss += loss.item()
            eot_loss += loss_components[0].item()
            trajectory_loss += loss_components[1].item()
            below_ground_loss += loss_components[2].item()

            if verbose:
                print(f"Avg loss for batch {idx + 1}: {loss.item():.8f}")

        avg_loss = total_loss / num_batches
        avg_eot_loss = eot_loss / num_batches
        avg_trajectory_loss = trajectory_loss / num_batches
        avg_below_ground_loss = below_ground_loss / num_batches

        if verbose:
            if valid:
                name = "validation"
            else:
                name = "test"
            print(f"Avg {name} loss: {avg_loss:.8f}")
            print(f"Avg {name} EOT loss: {avg_eot_loss:.8f}")
            print(f"Avg {name} trajectory loss: {avg_trajectory_loss:.8f}")
            print(f"Avg {name} below ground loss: {avg_below_ground_loss:.8f}")
        
        return avg_loss, (avg_eot_loss, avg_trajectory_loss, avg_below_ground_loss)
