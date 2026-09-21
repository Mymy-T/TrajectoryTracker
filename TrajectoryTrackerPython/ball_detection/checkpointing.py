import os
import torch
import settings

device = settings.device

def _compare_model_performance(checkpoint_model: tuple, new_model: tuple, precision_diff_threshold=0.01, min_recall=0.8): # TODO: Get baseline min_recall by comparing it to pretrained model performance, if it's not too low
    """
    checkpoint_model, new_model - tuples of (precision, mde, recall, epoch)
    Returns True is new model is better, False otherwise
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

# MARK: Checkpoint Manager
class CheckpointManager:

    def __init__(self, checkpoints_folder, max_checkpoints, best_model_metrics=None):
        self.checkpoints_folder = checkpoints_folder
        self.max_checkpoints = max_checkpoints  # None means keep every checkpoint
        self.best_model_metrics = best_model_metrics if best_model_metrics is not None else [] # List of (precision, mde, recall, epoch)

        if max_checkpoints is None:
            print(f"Keeping every checkpoint in {checkpoints_folder}")


    def update(self, model, epoch, precision, mde, recall):

        keep_all = self.max_checkpoints is None
        if keep_all or len(self.best_model_metrics) < self.max_checkpoints:
            self._save(model, epoch, precision, mde, recall)
            return

        worst_idx = 0
        for i in range(1, len(self.best_model_metrics)):
            if _compare_model_performance(checkpoint_model=self.best_model_metrics[i], new_model=self.best_model_metrics[worst_idx]): # If true, worst_idx doesn't refer to worst model checkpoint
                worst_idx = i

        if _compare_model_performance(checkpoint_model=self.best_model_metrics[worst_idx], new_model=(precision, mde, recall)):

            deleted_model_metrics = self.best_model_metrics.pop(worst_idx) # Delete worst checkpoint
            os.remove(self._path_for_epoch(deleted_model_metrics[-1]))
            
            self._save(model, epoch, precision, mde, recall) # Insert new checkpoint


    def best_checkpoint_path(self):

        best_idx = 0
        for i in range(1, len(self.best_model_metrics)):

            if _compare_model_performance(checkpoint_model=self.best_model_metrics[best_idx], new_model=self.best_model_metrics[i]): # If true, best_idx doesn't refer to best model checkpoint
                best_idx = i

        return self._path_for_epoch(self.best_model_metrics[best_idx][-1])


    def _path_for_epoch(self, epoch):
        return os.path.join(self.checkpoints_folder, f"epoch_{epoch}.pt")
    

    def _save(self, model, epoch, precision, mde, recall):
        self.best_model_metrics.append((precision, mde, recall, epoch))
        torch.save(model.state_dict(), self.__path_for_epoch(epoch))


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

        print(f"Resuming training from epoch {start_epoch}, starting with batch {resume_batch_idx + 1}") # Add one since batch idx is 0 indexed

        return {
            "best_model_metrics": resume_checkpoint["best_model_metrics"],
            "start_epoch": start_epoch,
            "resume_batch_idx": resume_batch_idx,
            "total_batch_loss": resume_checkpoint["total_batch_loss"],
        }


    def save(self, model, optimizer, epoch, resume_batch_idx, total_batch_loss, best_model_metrics):

        resume_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "resume_batch_idx": resume_batch_idx, # idx of the next batch, not the current completed batch
            "total_batch_loss": total_batch_loss, # For logging purposes
            "best_model_metrics": best_model_metrics,
        }

        torch.save(resume_checkpoint, self.resume_checkpoint_path)


    def delete(self):
        os.remove(self.resume_checkpoint_path) # Remove resume_checkpoint.pt, everything has been completed for this run
