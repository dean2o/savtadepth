import torch
import sys
from fastai.vision.all import *
from torchvision.utils import save_image


class ImageImageDataLoaders(DataLoaders):
    "Basic wrapper around several `DataLoader`s with factory methods for Image to Image problems"

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_label_func(cls, path, fnames, label_func, valid_pct=0.2, seed=None, item_tfms=None,
                        batch_tfms=None, **kwargs):
        "Create from list of `fnames` in `path`s with `label_func`."
        dblock = DataBlock(blocks=(ImageBlock(cls=PILImage), ImageBlock(cls=PILImageBW)),
                           splitter=RandomSplitter(valid_pct, seed=seed),
                           get_y=label_func,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        res = cls.from_dblock(dblock, fnames, path=path, **kwargs)
        return res


def get_y_fn(x):
    y = str(x.absolute()).replace('.jpg', '_depth.png')
    y = Path(y)

    return y


def create_data(data_path):
    fnames = get_files(data_path / 'train', extensions='.jpg')
    data = ImageImageDataLoaders.from_label_func(data_path / 'train', seed=42, bs=4, num_workers=0,
                                                 fnames=fnames, label_func=get_y_fn)
    return data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: %s <data_path>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    data = create_data(Path(sys.argv[1]))
    learner = unet_learner(data, resnet34, metrics=rmse, wd=1e-2, n_out=3, loss_func=MSELossFlat(),
                           path='src/test/')
    print("Training model...")
    learner.fine_tune(1)
    print("Saving model...")
    learner.save('model')
