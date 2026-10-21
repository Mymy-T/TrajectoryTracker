from ball_detection.model import TrackNet
from ball_detection.paths import ProjectPaths
from ball_detection import settings
from ball_detection.trainer import train_and_evaluate_model, RunType
import torch
import sys

# TODO: Save model after every single epoch; compare checkpoints at the end; 

if len(sys.argv) not in (3, 4):
    print(f"USAGE: python3 train_script.py <model_name> <resuming> [max_train_seconds]")
    sys.exit(1)
paths = ProjectPaths()
device = settings.device
model = TrackNet()
pretrained_state = torch.load(paths.pretrained_model_path, weights_only=True, map_location=torch.device(device))
model.load_state_dict(pretrained_state)
model.to(device)
name = sys.argv[1]
resuming = sys.argv[2].lower() == "true" # sys.argv values are always strings
max_train_seconds = int(sys.argv[3]) if len(sys.argv) == 4 else None
# NOTE: Test run; hparams aren't tuned yet, dataset isn't complete
result = train_and_evaluate_model(model=model, lr=1e-5, batch_size=4, eval_batch_size=8, accumulation_steps=8, max_checkpoints=None, max_epochs=10, name=name, resuming=resuming, run_type=RunType.final_model_training, verbose=True, max_train_seconds=max_train_seconds)
if result is None:
    print("Training paused before completion (time budget reached). Re-run with the same model name and resuming=True to continue.")
else:
    test_loss, precision, mde, recall = result
    mde_str = f"{mde:.4f}" if mde is not None else "N/A"
    print(f"Training completed successfully! Final stats:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"MDE: {mde_str}")
    print(f"Recall: {recall:.4f}")
