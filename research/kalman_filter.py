import numpy as np

dt = 1.0  # Time step

def predict(mu, sigma, A, R, B = 0, u = 0):
    # A = np.array([[1, dt], [0, 1]])  # State transition matrix
    mu_bar = A @ mu  + B * u
    sigma_bar = A @ sigma @ A.T + R
    return mu_bar, sigma_bar

def correct(mu_bar, sigma_bar, z, H, Q):
    # H = np.array([[1, 0]])  # Measurement matrix
    y = z - H @ mu_bar  # Innovation
    S = H @ sigma_bar @ H.T + Q  # Innovation covariance
    K = sigma_bar @ H.T @ np.linalg.inv(S)  # Kalman gain
    mu = mu_bar + K @ y
    sigma = (np.eye(len(mu)) - K @ H) @ sigma_bar
    return mu, sigma

def test(mu, sigma, A, R, H, Q, num_iterations=20, velocity=1):
    for t in range(num_iterations):
        true_position = velocity * dt * t  # ground truth
        z = true_position + np.random.normal()  # noisy measurement
        mu_bar, sigma_bar = predict(mu, sigma, A, R)
        mu, sigma = correct(mu_bar, sigma_bar, z, H, Q)
        print("Final estimated state (mu):", mu)
        print("Final estimated covariance (sigma):", sigma)
    return mu, sigma

test(np.array([[0], [1]]), np.eye(2), np.array([[1, dt], [0, 1]]), np.eye(2), np.array([[1, 0]]), np.eye(1), np.array([[1]]), 20, 1)
