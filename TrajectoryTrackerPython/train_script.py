import torch
from model import TrackNet
from paths import ProjectPaths
import sys
from trainer import train_and_evaluate_model, RunType
import train_settings

if len(sys.argv) != 2:
    print(f"USAGE: python3 train_script.py <model_name>")
    sys.exit(1)
paths = ProjectPaths()
device = train_settings.device
model = TrackNet()
pretrained_state = torch.load(paths.pretrained_model_path, weights_only=True, map_location=torch.device(device))
model.load_state_dict(pretrained_state)
model.to(device)
name = sys.argv[1]
# NOTE: Test run; hparams aren't tuned yet, dataset isn't complete
test_loss, precision, mde, recall = train_and_evaluate_model(model=model, lr=1e-5, batch_size=4, eval_batch_size=8, accumulation_steps=8, checkpoints=5, max_epochs=10, name=name, run_type=RunType.final_model_training, verbose=True)
mde_str = f"{mde:.4f}" if mde is not None else "N/A"
print(f"Training completed successfully! Final stats:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Precision: {precision:.4f}")
print(f"MDE: {mde_str}")
print(f"Recall: {recall:.4f}")
