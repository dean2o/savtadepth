"""Trains or fine-tunes a model for the task of monocular depth estimation
Receives 1 arguments from argparse:
  <data_path> - Path to the dataset which is split into 2 folders - train and test.
"""
import sys
from fastai.vision.all import unet_learner, Path, resnet34, rmse, MSELossFlat
from src.code.custom_data_loading import create_data
from dagshub.fastai import DAGsHubLogger


if __name__ == "__main__":
    # Check if got all needed input for argparse
    if len(sys.argv) != 2:
        print("usage: %s <data_path>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    data = create_data(Path(sys.argv[1]))
    wd, lr, ep = 1e-2, 1e-3, 1
    learner = unet_learner(data,
                           resnet34,
                           metrics=rmse,
                           wd=wd,
                           n_out=3,
                           loss_func=MSELossFlat(),
                           path='src/',
                           model_dir='models',
                           cbs=DAGsHubLogger(
                               metrics_path="logs/train_metrics.csv",
                               hparams_path="logs/train_params.yml"
                           ))

    print("Training model...")
    learner.fine_tune(epochs=ep, base_lr=lr)
    print("Saving model...")
    learner.save('model')
    print("Done!")
