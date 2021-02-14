from fastai.vision.all import \
    DataLoaders, \
    delegates, \
    DataBlock, \
    ImageBlock, \
    PILImage, \
    PILImageBW, \
    RandomSplitter, \
    Path, \
    get_files


class ImageImageDataLoaders(DataLoaders):
    """Basic wrapper around several `DataLoader`s with factory methods for Image to Image problems"""
    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_label_func(cls, path, filenames, label_func, valid_pct=0.2, seed=None, item_transforms=None,
                        batch_transforms=None, **kwargs):
        """Create from list of `fnames` in `path`s with `label_func`."""
        datablock = DataBlock(blocks=(ImageBlock(cls=PILImage), ImageBlock(cls=PILImageBW)),
                              get_y=label_func,
                              splitter=RandomSplitter(valid_pct, seed=seed),
                              item_tfms=item_transforms,
                              batch_tfms=batch_transforms)
        res = cls.from_dblock(datablock, filenames, path=path, **kwargs)
        return res


def get_y_fn(x):
    y = str(x.absolute()).replace('.jpg', '_depth.png')
    y = Path(y)

    return y


def create_data(data_path):
    filenames = get_files(data_path, extensions='.jpg')
    if len(filenames) == 0:
        raise ValueError("Could not find any files in the given path")
    dataset = ImageImageDataLoaders.from_label_func(data_path,
                                                    seed=42,
                                                    bs=4, num_workers=0,
                                                    filenames=filenames,
                                                    label_func=get_y_fn)
    return dataset
