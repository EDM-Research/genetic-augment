import numpy as np
import scipy.stats as stats


def wasserstein_distance(features: np.array, reference_features: np.array):
    """
    Compute the wasserstein distance between two arrays of multivariate samples
    Wasserstein distance is computed for each variable separately and averaged
    """
    distance_per_feature = []
    for i in range(features.shape[1]):
        distance_per_feature.append(stats.wasserstein_distance(reference_features[:, i], features[:, i]))

    return np.mean(distance_per_feature)


def variance(features: np.array):
    """
    Compute variance of array of multivariate samples
    Variance is computed for each sample and averaged
    """
    return np.mean(np.var(features, axis=0))
