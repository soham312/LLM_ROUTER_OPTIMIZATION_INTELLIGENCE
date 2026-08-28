import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

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

    Forced warm-start (bug fix, post-Stage 10a):
    A brand-new arm starts with theta_a = 0, so its expected_reward is exactly
    0.0 for every context until it has been updated at least once. Left alone,
    that's a deadlock: the caller-side confidence gate (OptimizationRouter's
    CONFIDENCE_THRESHOLD check) treats "expected_reward is 0" as "don't trust
    this pick" and routes to the fallback model instead - but an arm can only
    ever earn a non-zero expected_reward by actually being selected and
    updated. No arm but the (gate-exempt) fallback could ever be tried, so
    the router would converge to 100% fallback traffic regardless of
    context, decay, or embedding quality. We fix this the same way UCB1
    itself is defined: try every arm once, unconditionally, before letting
    UCB comparisons decide anything. `select_model` reports this via the
    `is_forced_exploration` flag so the router knows not to treat a
    warm-start pick's low expected_reward as low confidence.
    """

    def __init__(
        self,
        models: List[str],
        embedding_dim: int = 384,
        alpha: float = 1.0,
        gamma: float = 0.99,
        min_pulls_before_ucb: int = 1,
    ):
        """
        :param models: List of available model names (arms)
        :param embedding_dim: Dimension of the context vector
        :param alpha: Exploration parameter. Higher = more exploration.
        :param gamma: Decay factor for non-stationarity. 1.0 = no decay, < 1.0 = sliding window memory.
        :param min_pulls_before_ucb: Number of times each arm is forced to be
            tried (round-robin, in `models` order) before UCB comparisons are
            trusted to pick between arms. Must be >= 1 so every arm gets a
            real data point before it's judged against the others.
        """
        self.models = models
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.gamma = gamma
        self.min_pulls_before_ucb = max(1, min_pulls_before_ucb)

        # A_a: Covariance matrix for each arm. Initialized to Identity matrix.
        self.A = {m: np.eye(self.embedding_dim) for m in models}
        # A_inv: Inverse of A_a. Cached for performance.
        self.A_inv = {m: np.eye(self.embedding_dim) for m in models}
        # b_a: Reward vector for each arm. Initialized to zeros.
        self.b = {m: np.zeros(self.embedding_dim) for m in models}
        # Tracks how many real updates each arm has received, purely to
        # drive the warm-start decision below - independent of A/A_inv/b.
        self.pull_counts: Dict[str, int] = {m: 0 for m in models}

    def select_model(self, context: np.ndarray) -> Tuple[str, float, float, bool]:
        """
        Selects the best model to route the query to, based on the context vector.

        Exploration vs Exploitation:
        - The first term (context @ theta) is the Exploitation term (predicted reward).
        - The second term (alpha * sqrt(...)) is the Exploration term (uncertainty).
        Arms with high uncertainty will have a higher UCB and get explored.

        Returns:
            Tuple of (best_model, expected_reward, uncertainty, is_forced_exploration).
            `is_forced_exploration` is True when this pick is an unconditional
            warm-start trial rather than a genuine UCB comparison - callers
            should not treat its (necessarily low) expected_reward as a sign
            of low confidence.
        """
        # Ensure context is a 1D numpy array
        x = np.array(context).flatten()

        # Warm-start: any arm that hasn't yet met its minimum pull count is
        # tried unconditionally, in `models` order, before UCB drives the
        # decision at all.
        for model in self.models:
            if self.pull_counts[model] < self.min_pulls_before_ucb:
                theta = self.A_inv[model] @ self.b[model]
                expected_reward = x @ theta
                uncertainty = np.sqrt(x @ self.A_inv[model] @ x)
                return model, expected_reward, uncertainty, True

        best_model = None
        highest_ucb = -float('inf')
        best_exp_reward = 0.0
        best_uncertainty = 0.0

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

        return best_model, best_exp_reward, best_uncertainty, False

    def best_known_model(
        self,
        context: np.ndarray,
        exclude: Optional[str] = None,
        prefer: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Returns the arm with the highest current *exploitation-only*
        estimate (no exploration bonus) - used as an escalation target, not
        for normal routing. `select_model`'s UCB comparison is deliberately
        exploration-seeking (that's the whole point of a bandit); escalation
        wants the opposite question answered: "given everything we actually
        know right now, which arm looks best?"

        Why this exists (bug fix, post-Stage 10a): escalation used to always
        target a single hardcoded `fallback_model`, on the assumption that
        model is reliably trustworthy. If that specific model degrades
        (e.g. via judge.set_shock), the system had no way to notice and kept
        escalating right back into it. Consulting the bandit's own current
        belief instead lets escalation adapt when the "trusted" model's own
        track record has actually gotten worse.

        :param exclude: Never return this arm (e.g. the one that just
            triggered the escalation - retrying with itself is pointless).
        :param prefer: Tie-break preference. Early on, most/all arms are
            genuinely tied (e.g. all at exactly 0.0 immediately after
            warm-start), and picking a *consistent, configured* preference
            keeps behavior predictable during that bootstrap phase, while
            still letting real evidence override it the moment arms are no
            longer tied - the bandit is only ever blindly trusted here in
            the absence of any actual evidence to the contrary.
        """
        x = np.array(context).flatten()

        rewards: Dict[str, float] = {}
        for model in self.models:
            if model == exclude:
                continue
            theta = self.A_inv[model] @ self.b[model]
            rewards[model] = float(x @ theta)

        if not rewards:
            return None, 0.0

        best_reward = max(rewards.values())
        tied_for_best = [m for m, r in rewards.items() if r == best_reward]

        if prefer is not None and prefer in tied_for_best:
            return prefer, best_reward
        return tied_for_best[0], best_reward

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

        self.pull_counts[model] += 1

        logger.debug(f"Updated bandit for {model} with reward {reward:.4f}")
