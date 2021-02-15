import numpy as np


def compute_errors(target, prediction):
    thresh = np.maximum((target / prediction), (prediction / target))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(target - prediction) / target)
    sq_rel = np.mean(((target - prediction) ** 2) / target)

    rmse = (target - prediction) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(target) - np.log(prediction)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(prediction) - np.log(target)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(target) - np.log10(prediction))).mean()

    return a1, a2, a3, abs_rel, sq_rel, rmse, rmse_log, silog, log_10


def compute_eval_metrics(targets, predictions):
    targets = targets / 25.0
    predictions = predictions / 25.0

    min_depth_eval = 1e-3
    max_depth_eval = 10

    num_samples = predictions.shape[0]

    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    rmse = np.zeros(num_samples, np.float32)
    rmse_log = np.zeros(num_samples, np.float32)
    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):
        target_depth = targets[i]
        prediction_depth = predictions[i]

        prediction_depth[prediction_depth < min_depth_eval] = min_depth_eval
        prediction_depth[prediction_depth > max_depth_eval] = max_depth_eval
        prediction_depth[np.isinf(prediction_depth)] = max_depth_eval

        target_depth[np.isinf(target_depth)] = 0
        target_depth[np.isnan(target_depth)] = 0

        valid_mask = np.logical_and(target_depth > min_depth_eval, target_depth < max_depth_eval)

        a1[i], a2[i], a3[i], abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], silog[i], log10[i] = \
            compute_errors(target_depth[valid_mask], prediction_depth[valid_mask])

    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
    print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
        a1.mean(), a2.mean(), a3.mean(),
        abs_rel.mean(), sq_rel.mean(), rmse.mean(), rmse_log.mean(), silog.mean(), log10.mean()))

    return dict(a1=a1.mean(), a2=a2.mean(), a3=a3.mean(),
                abs_rel=abs_rel.mean(), sq_rel=sq_rel.mean(),
                rmse=rmse.mean(), rmse_log=rmse_log.mean(),
                log10=log10.mean(), silog=silog.mean())
