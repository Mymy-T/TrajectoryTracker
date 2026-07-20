import torch
from model import TrackNet
from paths import ProjectPaths
from trainer import train_and_evaluate_model, RunType
import train_settings


paths = ProjectPaths()
device = train_settings.device
model = TrackNet()
pretrained_state = torch.load(paths.pretrained_model_path, weights_only=True, map_location=torch.device(device))
model.load_state_dict(pretrained_state)
# NOTE: Test run; hparams aren't tuned yet, dataset isn't complete
test_loss = train_and_evaluate_model(model=model, lr=1e-5, batch_size=32, checkpoints=5, max_epochs=10, name="test_trial", run_type=RunType.final_model_training, verbose=True)
print(test_loss)
