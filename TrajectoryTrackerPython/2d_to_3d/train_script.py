import argparse
from model import TrajectoryReconstructionNetwork, FullBounceNetwork
from trainer import train_and_evaluate
import settings

def main():

    parser = argparse.ArgumentParser(description="Start/resume training a model.")
    parser.add_argument("--name", required=True, help="The name of the model. This should be a simple string, without extensions. Do NOT add _bm if it's a bounce model, it will be added automatically.")
    parser.add_argument("--type", required=True, choices=["full", "bounce"], help="Specify type of model to train.")
    parser.add_argument("--resuming", action="store_true", help="Whether or not this training session should resume a halfway-trained model.")
    parser.add_argument("--maxtime", type=int, help="Max time before training will stop (in SECONDS). No time limit if not provided.")
    parser.add_argument("--maxcheckpoints", type=int, help="Max checkpoints of models to save before the worst performing one will be deleted upon completion of a training loop step. No limit if not provided.")

    args = parser.parse_args()

    device = settings.device

    name = args.name

    if args.type == "full":
        model = TrajectoryReconstructionNetwork(saved_bounce_model_name=f"{name}_bm")
    elif args.type == "bounce":
        model = FullBounceNetwork()

    model.to(device)

    train_config = settings.Training().train_config
    train_config.name = name
    train_config.resuming = args.resuming
    train_config.max_train_seconds = args.maxtime
    train_config.max_checkpoints = args.maxcheckpoints

    result = train_and_evaluate(model=model, train_config=train_config, verbose=True)
    if result is None:
        print("Training paused before completion (time budget reached). Re-run with the same model name and resuming=True to continue.")
    else:
        test_loss, test_loss_components = result
        print(f"Training completed successfully! Final stats:")
        print(f"Test Loss: {test_loss:.4f}")
        for key, value in test_loss_components.items():
            print(f"{key} Component: {value:.4f}")

if __name__ == "__main__":
    main()