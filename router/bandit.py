import numpy as np
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class LinUCBRouter:
    """
    Continuous Contextual Bandit using LinUCB with exponential decay.
    
    Why LinUCB?
    We have continuous context vectors (embeddings) representing user queries. 
    LinUCB assumes the expected reward of a model (arm) is a linear function 
    of the context. It balances exploration (trying models we are uncertain about) 
    and exploitation (using the best known model) in a principled way.
    
    Stage 3 & 4 implementation details:
    - Math: For each arm 'a', we maintain a matrix A_a (dimension d x d) and 
      a vector b_a (dimension d). The estimated coefficient theta_a = inv(A_a) @ b_a.
      The upper confidence bound (UCB) is calculated as:
      score_a = context @ theta_a + alpha * sqrt(context @ inv(A_a) @ context)
    - Decay (Stage 4): To handle non-stationary environments (e.g. model degrades, 
      or pricing changes mid-run), we apply an exponential decay factor (gamma) 
      to A_a and b_a during updates:
      A_a = gamma * A_a + context @ context^T
      b_a = gamma * b_a + reward * context
      This gradually forgets old observations, allowing the bandit to adapt to 'shocks'.
    """
    
    def __init__(self, models: List[str], embedding_dim: int = 384, alpha: float = 1.0, gamma: float = 0.99):
        """
        :param models: List of available model names (arms)
        :param embedding_dim: Dimension of the context vector
        :param alpha: Exploration parameter. Higher = more exploration.
        :param gamma: Decay factor for non-stationarity. 1.0 = no decay, < 1.0 = sliding window memory.
        """
        self.models = models
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.gamma = gamma
        
        # A_a: Covariance matrix for each arm. Initialized to Identity matrix.
        self.A = {m: np.eye(self.embedding_dim) for m in models}
        # A_inv: Inverse of A_a. Cached for performance.
        self.A_inv = {m: np.eye(self.embedding_dim) for m in models}
        # b_a: Reward vector for each arm. Initialized to zeros.
        self.b = {m: np.zeros(self.embedding_dim) for m in models}
        
    def select_model(self, context: np.ndarray) -> Tuple[str, float, float]:
        """
        Selects the best model to route the query to, based on the context vector.
        
        Exploration vs Exploitation:
        - The first term (context @ theta) is the Exploitation term (predicted reward).
        - The second term (alpha * sqrt(...)) is the Exploration term (uncertainty).
        Arms with high uncertainty will have a higher UCB and get explored.
        
        Returns:
            Tuple of (best_model, expected_reward, uncertainty) for the chosen model.
            This allows the router to trigger escalation if expected reward/confidence is too low.
        """
        best_model = None
        highest_ucb = -float('inf')
        best_exp_reward = 0.0
        best_uncertainty = 0.0
        
        # Ensure context is a 1D numpy array
        x = np.array(context).flatten()
        
        for model in self.models:
            # theta_a = A_a^(-1) @ b_a
            theta = self.A_inv[model] @ self.b[model]
            
            # Exploitation: Expected reward
            expected_reward = x @ theta
            
            # Exploration: Confidence interval size
            # sqrt(x^T * A_inv * x)
            uncertainty = np.sqrt(x @ self.A_inv[model] @ x)
            
            # Upper Confidence Bound
            ucb = expected_reward + self.alpha * uncertainty
            
            if ucb > highest_ucb:
                highest_ucb = ucb
                best_model = model
                best_exp_reward = expected_reward
                best_uncertainty = uncertainty
                
        return best_model, best_exp_reward, best_uncertainty

    def update(self, model: str, context: np.ndarray, reward: float):
        """
        Updates the bandit's internal state with the observed reward.
        Applies exponential decay to handle non-stationarity.
        """
        if model not in self.models:
            logger.warning(f"Attempted to update unknown model '{model}'. Ignoring.")
            return
            
        x = np.array(context).flatten()
        
        # Apply exponential decay and update
        # A_a = gamma * A_a + x * x^T
        self.A[model] = self.gamma * self.A[model] + np.outer(x, x)
        
        # b_a = gamma * b_a + r * x
        self.b[model] = self.gamma * self.b[model] + reward * x
        
        # Recompute inverse (using Sherman-Morrison could be faster, but direct inv is fine for dim=384)
        # We add a small ridge (e.g., 1e-4 * I) before inverting for numerical stability
        self.A_inv[model] = np.linalg.inv(self.A[model] + 1e-4 * np.eye(self.embedding_dim))
        
        logger.debug(f"Updated bandit for {model} with reward {reward:.4f}")
