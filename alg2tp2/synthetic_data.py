from sklearn.datasets import make_blobs
import numpy as np

def generate_synthetic_data_blobs(n_samples=1000, centers=3, cluster_std=1.0, random_state=42):
    X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)
    return X

def generate_synthetic_data_gaussian(n_samples=1000, centers=3, std_range=(0.1, 1.0)):
    means = [np.random.rand(2) * 10 for _ in range(centers)]
    std_devs = np.random.uniform(std_range[0], std_range[1], centers)
    X = np.vstack([np.random.multivariate_normal(mean, np.eye(2) * std_dev, n_samples // centers)
                   for mean, std_dev in zip(means, std_devs)])
    return X
