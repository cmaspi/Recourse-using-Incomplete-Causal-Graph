import numpy as np
from dataset_generator import Diamond
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

    return test_stat < 3 * thresh


DataGen = Diamond(
    lambda: np.random.uniform(-5, 5),
    lambda: np.random.normal(0, 1),
    lambda: np.random.normal(0, 1),
    lambda: np.random.normal(0, 1),
)

data = DataGen.generate_diamond(10000)

# Lets assume that we use pc algorithm to find the edges in causal graph
cg = [(0, 1), (1, 0), (1, 3), (3, 1), (3, 2), (2, 3), (0, 2), (2, 0)]
causal_graph = []
for x, y in cg:
    if is_consistent(data, x, y):
        causal_graph.append((x, y))

print(causal_graph)
