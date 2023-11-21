import numpy as np
from dataset_generator import *
from sklearn.svm import SVR
from hsic import hsic_gam
from sklearn.model_selection import train_test_split


def is_consistent(data, x, y):
    X = data[:, x].reshape(-1, 1)
    y = data[:, y]

    # Fitting a non-linear regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    svr = SVR(kernel="rbf", gamma="auto")
    svr.fit(X_train, y_train)

    y_pred = svr.predict(X_test)
    residuals = (y_test - y_pred).reshape(-1, 1)

    test_stat, thresh = hsic_gam(X_test, residuals, 0.05)

    return test_stat < 3*thresh


if __name__ == "__main__":
    DataGen = GermanCredit(
        lambda: np.random.binomial(n=1, p=0.5),
        lambda: np.random.gamma(10, 3.5),
        lambda: np.random.normal(0, 0.5),
        lambda: np.random.normal(0, 2),
        lambda: np.random.normal(0, 3),
        lambda: np.random.normal(0, 2),
        lambda: np.random.normal(0, 5),
    )

    data = DataGen.generate_german_credit(5000)

    cg = [(5, 0), (2, 1), (3, 1), (4, 1), (5, 1), (0, 2), (1, 2), (1, 3), (4, 3), (0, 4), (1, 4), (3, 4), (0, 6), (5, 6), (2, 0), (4, 0), (6, 0), (0, 5), (1, 5), (6, 5), (2, 5), (5, 2)]
    causal_graph = []
    for x, y in cg:
        if is_consistent(data, x, y):
            causal_graph.append((x, y))

    print(causal_graph)
