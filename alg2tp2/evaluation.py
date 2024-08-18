from sklearn.metrics import silhouette_score, adjusted_rand_score

def evaluate_clustering(X, labels, true_labels=None):
    silhouette = silhouette_score(X, labels)
    rand_index = adjusted_rand_score(true_labels, labels) if true_labels is not None else None
    return silhouette, rand_index
