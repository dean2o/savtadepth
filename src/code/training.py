import torch
import sys
from fastai2.vision.all import *
from torchvision.utils import save_image


def get_y_fn(x):
    y = str(x.absolute()).replace('.jpg', '_depth.png')
    y = Path(y)

    return y


def create_data(data_path):
    fnames = get_files(data_path/'train', extensions='.jpg')
    data = SegmentationDataLoaders.from_label_func(data_path/'train',  bs=4, num_workers=0, fnames=fnames, label_func=get_y_fn)
    return data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: %s <data_path>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    data = create_data(Path(sys.argv[1]))
    learner = unet_learner(data, resnet34, metrics=rmse, wd=1e-2, n_out=1, loss_func=MSELossFlat(), path='src/')
    learner.fine_tune(1)

    learner.save('model')
