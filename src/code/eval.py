import sys
from fastai.vision.all import unet_learner, Path, resnet34, MSELossFlat, get_files, L, tuplify
from src.code.custom_data_loading import create_data
from src.code.eval_metric_calculation import compute_eval_metrics
from dagshub import dagshub_logger


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: %s <test_data_path>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    data_path = Path(sys.argv[1])
    data = create_data(data_path)

    filenames = get_files(Path(sys.argv[1]), extensions='.jpg')
    test_files = L([Path(i) for i in filenames])
    test_dl = data.test_dl(test_files, with_labels=True)
    learner = unet_learner(data,
                           resnet34,
                           n_out=3,
                           loss_func=MSELossFlat(),
                           path='src/',
                           model_dir='models')
    learner = learner.load('model')
    inputs, predictions, targets, decoded = learner.get_preds(dl=test_dl,
                                                              with_input=True,
                                                              with_decoded=True)
    # FastAI magic to retrieve image values
    inputs = (inputs,)
    decoded_predictions = learner.dls.decode(inputs + tuplify(decoded))[1]
    decoded_targets = learner.dls.decode(inputs + tuplify(targets))[1]

    metrics = compute_eval_metrics(decoded_targets.numpy(), decoded_predictions.numpy())

    with dagshub_logger(
            metrics_path="logs/test_metrics.csv",
            should_log_hparams=False
    ) as logger:
        # Metric logging
        logger.log_metrics(metrics)

    print("Evaluation Done!")
