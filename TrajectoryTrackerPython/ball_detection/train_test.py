import random

from torch.utils.data import DataLoader, Subset

from dataset import DatasetType, TrackNetDataset
from model import TrackNet
import trainer
import train_settings

device = train_settings.device


def _make_subset(dataset_type, subset_size, seed):
    dataset = TrackNetDataset(dataset_type)
    n = min(subset_size, len(dataset))
    indices = random.Random(seed).sample(range(len(dataset)), k=n)
    return Subset(dataset, indices)


def _print_confusion_matrix(tp, tn, fp, fn):
    print("\nConfusion matrix (ball presence):")
    print(f"{'':20}{'Pred: ball':>15}{'Pred: no ball':>15}")
    print(f"{'Actual: ball':20}{tp:>15}{fn:>15}")
    print(f"{'Actual: no ball':20}{fp:>15}{tn:>15}")


def _print_metrics(avg_loss, tp, tn, fp, fn, error_sum):
    precision, mde, recall = trainer.compute_eval_metrics(tp, tn, fp, fn, error_sum)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mde_str = f"{mde:.4f}" if mde is not None else "N/A"

    print("\n===== Evaluation metrics (test subset) =====")
    print(f"Avg loss:  {avg_loss:.6f}")
    _print_confusion_matrix(tp, tn, fp, fn)
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"MDE:       {mde_str}")


def run_train_test(model=None, loss_fn=None, train_subset_size=64, eval_subset_size=32, batch_size=4, eval_batch_size=None, accumulation_steps=1, epochs=3, lr=1e-4, seed=0, verbose=True):
    """
    Trains <model> for a few epochs on a small random subset of the real dataset, then prints
    evaluation metrics (confusion matrix, precision, recall, F1, accuracy, MDE) on a small test
    subset. Nothing is logged to TensorBoard or saved to disk. Intended for quickly sanity-checking
    a training technique (e.g. a new loss function or architecture tweak) without needing a full,
    high-compute training run.
    """
    model = model if model is not None else TrackNet()
    loss_fn = loss_fn if loss_fn is not None else trainer.FocalLoss(gamma=2.0)
    model.to(device)

    train_subset = _make_subset(DatasetType.train, train_subset_size, seed)
    valid_subset = _make_subset(DatasetType.valid, eval_subset_size, seed)
    test_subset = _make_subset(DatasetType.test, eval_subset_size, seed)

    trainer.train_and_evaluate_model(
        model=model,
        lr=lr,
        batch_size=batch_size,
        max_checkpoints=None,
        max_epochs=epochs,
        name="train_test",
        resuming=False,
        run_type=trainer.RunType.final_model_training,
        verbose=verbose,
        eval_batch_size=eval_batch_size,
        accumulation_steps=accumulation_steps,
        log_and_save=False,
        train_dataset=train_subset,
        valid_dataset=valid_subset,
        test_dataset=test_subset,
    )
    # model was trained in place above regardless of the return value, since log_and_save=False
    # skips train_and_evaluate_model's "reload best checkpoint" step.

    test_loader = DataLoader(test_subset, batch_size=eval_batch_size or batch_size)
    avg_loss, tp, tn, fp, fn, error_sum = trainer.evaluate(model, loss_fn, test_loader, valid=False, verbose=False)
    _print_metrics(avg_loss, tp, tn, fp, fn, error_sum)

    return model


if __name__ == "__main__":
    # Edit the model / loss_fn below to try a different architecture or loss function.
    run_train_test(
        model=TrackNet(),
        loss_fn=trainer.FocalLoss(gamma=2.0),
        train_subset_size=64,
        eval_subset_size=32,
        batch_size=4,
        accumulation_steps=1,
        epochs=3,
        lr=1e-4,
    )
