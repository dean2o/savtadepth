import sys
import yaml
from fastai.vision.all import unet_learner, Path, resnet34, MSELossFlat, tuplify
from custom_data_loading import create_data
from eval_metric_calculation import compute_eval_metrics
from dagshub import dagshub_logger


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: %s <test_data_path>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    with open(r"./src/code/params.yml") as f:
        params = yaml.safe_load(f)

    data_path = Path(sys.argv[1])
    data, test_dl = create_data(data_path, is_test=True)

    arch = {'resnet34': resnet34}
    loss = {'MSELossFlat': MSELossFlat()}

    learner = unet_learner(data,
                           arch.get(params['architecture']),
                           n_out=int(params['num_outs']),
                           loss_func=loss.get(params['loss_func']),
                           path='src/',
                           model_dir='models')
    learner = learner.load('model')

    print("Running model on test data...")
    inputs, predictions, targets, decoded = learner.get_preds(dl=test_dl,
                                                              with_input=True,
                                                              with_decoded=True)
    # FastAI magic to retrieve image values
    inputs = (inputs,)
    decoded_predictions = learner.dls.decode(inputs + tuplify(decoded))[1]
    decoded_targets = learner.dls.decode(inputs + tuplify(targets))[1]

    print("Calculating metrics...")
    metrics = compute_eval_metrics(decoded_targets.numpy(), decoded_predictions.numpy())

    with dagshub_logger(
            metrics_path="logs/test_metrics.csv",
            should_log_hparams=False
    ) as logger:
        # Metric logging
        logger.log_metrics(metrics)

    print("Evaluation Done!")
