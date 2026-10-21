from convert_2d_to_3d.model import TypedModel, ModelType
from convert_2d_to_3d import settings
import os
import pandas as pd
import torch

device = settings.device

def _compare_model_performance(checkpoint_model: tuple, new_model: tuple):
    """
    checkpoint_model, new_model - tuples of (loss, epoch, full_loss) where full_loss is (total_loss, loss_components)
    Returns True is new model is better, False otherwise
    """
    return new_model[0] < checkpoint_model[0]

def _get_selection_loss(model: TypedModel, total_loss, loss_components):
    """
    Return the scalar loss used for comparing model performance
    """
    match model.model_type:
        case ModelType.FullModel:
            selection_loss = loss_components["Trajectory Loss"]
        case ModelType.BounceModel:
            selection_loss = total_loss
    
    return selection_loss

# MARK: Checkpoint Manager
class CheckpointManager:

    def __init__(self, checkpoints_folder, max_checkpoints, best_model_metrics=None):
        self.checkpoints_folder = checkpoints_folder
        self.max_checkpoints = max_checkpoints  # None means keep every checkpoint
        self.best_model_metrics = best_model_metrics if best_model_metrics is not None else [] # List of (loss, epoch, full_loss); loss is selection criteria, full_loss is tuple of (total loss, loss_components)

        if max_checkpoints is None:
            print(f"Keeping every checkpoint in {checkpoints_folder}")


    def update(self, model: TypedModel, epoch, loss, loss_components):

        keep_all = self.max_checkpoints is None
        if keep_all or len(self.best_model_metrics) < self.max_checkpoints:
            self._save(model, epoch, loss, loss_components)
            return
        
        selection_loss = _get_selection_loss(model, loss, loss_components)

        worst_idx = 0
        for i in range(1, len(self.best_model_metrics)):
            if _compare_model_performance(checkpoint_model=self.best_model_metrics[i], new_model=self.best_model_metrics[worst_idx]): # If true, worst_idx doesn't refer to worst model checkpoint
                worst_idx = i

        if _compare_model_performance(checkpoint_model=self.best_model_metrics[worst_idx], new_model=(selection_loss, epoch, (loss, loss_components))):
            deleted_model_metrics = self.best_model_metrics.pop(worst_idx) # Delete worst checkpoint
            os.remove(self._path_for_epoch(deleted_model_metrics[1]))
            self._save(model, epoch, loss, loss_components) # Insert new checkpoint


    def best_checkpoint_path(self):
        best_idx = 0

        for i in range(1, len(self.best_model_metrics)):
            if _compare_model_performance(checkpoint_model=self.best_model_metrics[best_idx], new_model=self.best_model_metrics[i]): # If true, best_idx doesn't refer to best model checkpoint
                best_idx = i

        return self._path_for_epoch(self.best_model_metrics[best_idx][1])
    

    def save_final_checkpoint_stats(self):

        save_path = os.path.join(self.checkpoints_folder, "checkpoint_stats.csv")
        df = pd.DataFrame()

        for model_metrics in self.best_model_metrics:
            epoch = model_metrics[1]
            total_loss = model_metrics[2][0]
            loss_components = model_metrics[2][1]

            row = pd.DataFrame([{
                "Epoch": epoch,
                "Total Loss": total_loss,
                **loss_components
            }])

            df = pd.concat([df, row], ignore_index=True)
        
        df = df.sort_values("Epoch")
        df.to_csv(save_path, index=False)


    def _path_for_epoch(self, epoch):
        return os.path.join(self.checkpoints_folder, f"epoch_{epoch}.pt")


    def _save(self, model: TypedModel, epoch, loss, loss_components):

        selection_loss = _get_selection_loss(model, loss, loss_components)

        self.best_model_metrics.append((selection_loss, epoch, (loss, loss_components)))
        torch.save(model.state_dict(), self._path_for_epoch(epoch))


# MARK: Resume Manager
class ResumeCheckpointManager:

    def __init__(self, resume_checkpoint_path):
        self.resume_checkpoint_path = resume_checkpoint_path

    def load(self, model, optimizer):

        if not os.path.exists(self.resume_checkpoint_path):
            raise FileNotFoundError(f"There is no existing resume_checkpoint.pt file. Ensure it is named correctly and at the correct path ({self.resume_checkpoint_path}) or run train_script.py with the resuming arg set to false.")
        
        resume_checkpoint = torch.load(self.resume_checkpoint_path, map_location=device)

        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = resume_checkpoint["epoch"]
        resume_batch_idx = resume_checkpoint["resume_batch_idx"] # idx of batch that training will start on
        best_model_metrics = resume_checkpoint["best_model_metrics"]
        loss_log = resume_checkpoint["loss_info"]

        print(f"Resuming training from epoch {start_epoch}, starting with batch {resume_batch_idx + 1}") # Add one since batch idx is 0 indexed

        return {
            "best_model_metrics": best_model_metrics,
            "start_epoch": start_epoch,
            "resume_batch_idx": resume_batch_idx,
            "loss_info": loss_log
        }


    def save(self, model, optimizer, epoch, resume_batch_idx, loss_log, best_model_metrics):

        resume_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "resume_batch_idx": resume_batch_idx, # idx of the next batch, not the current completed batch
            "loss_info": loss_log, # For logging purposes
            "best_model_metrics": best_model_metrics,
        }

        torch.save(resume_checkpoint, self.resume_checkpoint_path)


    def delete(self):
        os.remove(self.resume_checkpoint_path) # Remove resume_checkpoint.pt, everything has been completed for this run


# MARK: Loss Log
class LossLog:

    def __init__(self, total_loss: float, loss_components: dict):
        self.total_loss = total_loss
        self.loss_components = loss_components
    

    def get_loss_dict(self):
        return {
            "total_loss": self.total_loss,
            **self.loss_components
        }

