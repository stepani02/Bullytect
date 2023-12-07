import numpy as np

class SVM:
    def __init__(self, C=1.0, tol=1e-3, max_iter=100):
        self.C = C  # Regularization parameter
        self.tol = tol  # Tolerance for stopping criterion
        self.max_iter = max_iter  # Maximum number of iterations
        self.alpha = None  # Lagrange multipliers
        self.b = 0  # Bias term
        self.support_vectors = None
        self.support_vector_labels = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

    # Initialize Lagrange multipliers and bias
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        for _ in range(self.max_iter):
            for i in range(n_samples):
                # Compute predicted class and error
                f_i = np.dot(self.alpha * y, np.dot(X, X[i])) + self.b
                E_i = f_i - y[i]

                # Check the sample
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    # Select a random second sample j, different from i
                    j = np.random.choice(list(range(n_samples - 1)))
                    j = j + 1 if j >= i else j

                    # Compute predicted class and error for the second sample
                    f_j = np.dot(self.alpha * y, np.dot(X, X[j])) + self.b
                    E_j = f_j - y[j]

                # Save old values of alpha
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                # Compute the bounds L and H
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                # Check if L and H are equal
                    if L == H:
                        continue

                # Compute the second derivative of the objective function
                    eta = 2 * np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])

                # Skip the iteration if the second derivative is non-positive
                    if eta >= 0:
                        continue

                    # Update alpha[j]
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta

                    # Clip alpha[j] to the interval [L, H]
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    # Check if alpha[j] has changed significantly
                    if np.abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha[i] based on the change in alpha[j]
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    # Compute bias terms
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * np.dot(X[i], X[i]) - \
                        y[j] * (self.alpha[j] - alpha_j_old) * np.dot(X[i], X[j])
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * np.dot(X[i], X[j]) - \
                        y[j] * (self.alpha[j] - alpha_j_old) * np.dot(X[j], X[j])

                    # Update bias term
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

        # Support vectors have non-zero Lagrange multipliers
        support_vector_mask = self.alpha > 1e-5
        self.support_vectors = X[support_vector_mask]
        self.support_vector_labels = y[support_vector_mask]
        self.alpha = self.alpha[support_vector_mask]

        if len(self.support_vectors) == 0:
            raise ValueError("No support vectors found. Consider adjusting parameters or preprocessing the data.")

    def predict(self, X):
        # Ensure that X has the same number of features as the training data
        if X.shape[1] != self.support_vectors.shape[1]:
            raise ValueError("Number of features in X must match the number of features in the training data.")

    # Transpose self.support_vectors for correct dot product
        support_vectors_transposed = self.support_vectors.T

    # Ensure that self.alpha and self.support_vector_labels have the same shape
        self.alpha = self.alpha[:min(len(self.alpha), len(self.support_vector_labels))]
        self.support_vector_labels = self.support_vector_labels[:min(len(self.alpha), len(self.support_vector_labels))]

    # Ensure that X and support_vectors_transposed have the same number of features
        X = X[:, :self.support_vectors.shape[1]]

    # Compute the decision function using einsum
        decision_function = np.dot(X, support_vectors_transposed)  

    # Return the predicted class labels (1 or -1)
        return np.sign(np.dot(self.alpha * self.support_vector_labels, decision_function.T) + self.b).astype(int)