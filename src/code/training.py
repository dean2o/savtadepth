import torch
import sys
from fastai.vision import unet_learner, ImageImageList, models, Path, root_mean_squared_error


def get_y_fn(x):
    y = str(x.absolute()).replace('.jpg', '_depth.png')
    y = Path(y)

    return y


def create_databunch(data_path):
    data = (ImageImageList.from_folder(data_path)
            .filter_by_func(lambda fname: fname.suffix == '.jpg')
            .split_by_folder(train='train', valid='test')
            .label_from_func(get_y_fn).databunch()).normalize()
    return data


def train(data):
    learner = unet_learner(data, models.resnet34, metrics=root_mean_squared_error, wd=1e-2, loss_func=torch.nn.SmoothL1Loss())
    learner.fit_one_cycle(1, 1e-3)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: %s <data_path> <out_folder>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    data = create_databunch(sys.argv[1])
    data.batch_size = 1
    data.num_workers = 0
    learner = train(data)

    learner.save(sys.argv[2])
    learner.show_results()
