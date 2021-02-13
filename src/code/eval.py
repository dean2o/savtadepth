import sys
from fastai.vision.all import unet_learner, Path, resnet34, MSELossFlat
import torch
from src.code.custom_data_loading import create_data
from dagshub.fastai import DAGsHubLogger


def compute_errors(targ, pred):
    thresh = torch.max((targ / pred), (pred / targ)).numpy()
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = (torch.abs(targ - pred) / targ).mean().item()
    sq_rel = torch.mean(((targ - pred).pow(2)) / targ).item()

    rmse = torch.sqrt((targ - pred).pow(2).mean()).item()

    rmse_log = torch.sqrt((torch.log(1 + targ) - torch.log(1 + pred)).pow(2).mean()).item()

    err = torch.log(1 + pred) - torch.log(1 + targ)
    silog = torch.sqrt(torch.mean(err.pow(2)) - torch.mean(err).pow(2)).item() * 100

    log_10 = (torch.abs(torch.log10(1 + targ) - torch.log10(1 + pred))).mean().item()
    return dict(a1=a1,
                a2=a2,
                a3=a3,
                abs_rel=abs_rel,
                sq_rel=sq_rel,
                rmse=rmse,
                rmse_log=rmse_log,
                silog=silog,
                log_10=log_10)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: %s <test_data_path>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    data_path = Path(sys.argv[1])
    data = create_data(data_path, is_test=True)

    learner = unet_learner(data,
                           resnet34,
                           n_out=3,
                           loss_func=MSELossFlat(),
                           path='src/',
                           model_dir='models')
    learner = learner.load('model')
    predictions, targets = learner.get_preds()
    print(compute_errors(targets, predictions))
