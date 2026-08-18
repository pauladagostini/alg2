from alg2tp2.data import generate_blobs, generate_gaussian_mixture


def test_generate_blobs_shapes_match():
    X, y = generate_blobs(n_samples=90, centers=3, random_state=0)
    assert X.shape == (90, 2)
    assert y.shape == (90,)
    assert len(set(y)) == 3


def test_generate_gaussian_mixture_shapes_match():
    X, y = generate_gaussian_mixture(n_samples=90, centers=3, random_state=0)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 2
