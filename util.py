import numpy as np

"""
Split feature folds and target folds into training and testing sets.
One fold becomes the test set, the rest become the training set.
"""
def training_testing_split(feature_folds, target_folds, test_fold, validation_fold=None):
    test_features = feature_folds[test_fold]
    test_targets = target_folds[test_fold]
    train_features = np.empty(shape=(0, test_features.shape[1]))
    train_targets = np.array([])
    for i in range(len(feature_folds)):
        if i == test_fold:
            continue
        if i == validation_fold:
            continue
        train_features = np.append(train_features, feature_folds[i], axis=0)
        train_targets = np.append(train_targets, target_folds[i])
    if validation_fold is None:
        return train_features, train_targets, test_features, test_targets
    else:
        validation_features = feature_folds[validation_fold]
        validation_targets = target_folds[validation_fold]
        return train_features, train_targets, validation_features, validation_targets, test_features, test_targets


def split_n_folds(features, targets, n_folds, dummy=False):
    feature_folds = [np.empty(shape=(0, features.shape[1]), dtype=np.int32) for i in range(n_folds)]
    target_folds = [np.empty(shape=(0, targets.shape[0]), dtype=np.int32) for i in range(n_folds)]
    first_datapoint = 0
    num_datapoints = targets.shape[0]

    dummy_target = np.ones(shape=(1, features.shape[1]))
    dummy_feature = np.array([21])

    for i in range(n_folds):
        last_datapoint = int(num_datapoints / 10 * (i + 1))
        feature_folds[i] = features[first_datapoint:last_datapoint, :]
        target_folds[i] = targets[first_datapoint:last_datapoint]
        if dummy:
            feature_folds[i] = np.append(feature_folds[i], dummy_target, axis=0)
            target_folds[i] = np.append(target_folds[i], dummy_feature)
        first_datapoint = last_datapoint
    return feature_folds, target_folds