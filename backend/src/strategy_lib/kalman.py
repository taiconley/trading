import numpy as np
from typing import Tuple, Optional

class KalmanFilter:
    """
    A lightweight Kalman Filter implementation for pairs trading.
    Estimates the hedge ratio (beta) and intercept (alpha) dynamically.
    
    State vector x = [beta, alpha]^T
    Observation y = price_a
    Observation matrix H = [price_b, 1]
    
    Model:
    x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)  (State transition)
    y_t = H_t x_t + v_t,  v_t ~ N(0, R)  (Observation)
    """
    
    def __init__(self, delta: float = 1e-4, R: float = 1e-3):
        """
        Initialize the Kalman Filter.
        
        Args:
            delta: System noise covariance scaling factor (controls how fast parameters adapt)
            R: Measurement noise variance (controls sensitivity to noise)
        """
        self.n_states = 2
        
        # State vector [beta, alpha]
        self.x = np.zeros(self.n_states)
        
        # State covariance matrix
        self.P = np.eye(self.n_states)
        
        # System noise covariance (Q)
        # We assume random walk for parameters
        self.Q = np.eye(self.n_states) * delta
        
        # Measurement noise variance (R)
        self.R = R
        
        self.is_initialized = False

    def update(self, price_a: float, price_b: float) -> Tuple[float, float, float]:
        """
        Update the Kalman Filter with new observations and return the spread (prediction error).
        
        Args:
            price_a: Price of the dependent asset (y)
            price_b: Price of the independent asset (x)
            
        Returns:
            Tuple of (spread, beta, alpha)
            spread is the prediction error: y - (beta * x + alpha)
        """
        # Observation matrix H = [price_b, 1]
        H = np.array([price_b, 1.0])
        
        if not self.is_initialized:
            # Initialize state with first observation
            # Assume beta=1, alpha=price_a - price_b as a rough start
            self.x = np.array([1.0, price_a - price_b])
            self.is_initialized = True
            return 0.0, 1.0, price_a - price_b

        # 1. Prediction Step
        # State prediction: x_{t|t-1} = x_{t-1|t-1} (Random walk model)
        x_pred = self.x
        # Covariance prediction: P_{t|t-1} = P_{t-1|t-1} + Q
        P_pred = self.P + self.Q
        
        # 2. Update Step
        # Innovation (prediction error or "spread"): y - H * x_pred
        y_pred = np.dot(H, x_pred)
        error = price_a - y_pred
        
        # Innovation covariance: S = H * P_pred * H^T + R
        S = np.dot(H, np.dot(P_pred, H.T)) + self.R
        
        # Kalman Gain: K = P_pred * H^T * S^-1
        K = np.dot(P_pred, H.T) / S
        
        # Update state: x_{t|t} = x_{t|t-1} + K * error
        self.x = x_pred + K * error
        
        # Update covariance: P_{t|t} = (I - K * H) * P_{t|t-1}
        # Using Joseph form for numerical stability: P = (I-KH)P(I-KH)' + KRK'
        I = np.eye(self.n_states)
        I_KH = I - np.outer(K, H)
        self.P = np.dot(I_KH, np.dot(P_pred, I_KH.T)) + np.outer(K, K) * self.R
        
        beta, alpha = self.x[0], self.x[1]
        
        return error, beta, alpha

    def get_state(self) -> Tuple[float, float]:
        """Return current beta and alpha."""
        return self.x[0], self.x[1]
