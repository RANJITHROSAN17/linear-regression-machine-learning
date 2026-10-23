from salary_prediction import load_data, train


def test_data_shape():
    X, y = load_data()
    assert X.shape == (5000, 1) and y.shape == (5000,)


def test_model_fits_well():
    X, y = load_data()
    _, metrics, _ = train(X, y)
    assert metrics["r2_test"] > 0.95
    assert metrics["coef"] > 0  # salary grows with experience
