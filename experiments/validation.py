import numpy as np
import logging
from typing import List, Dict, Callable
from router.router_core import OptimizationRouter

logger = logging.getLogger(__name__)

class ValidationHarness:
    """
    STAGE 7: Validation Harness for Bandit Routing.
    
    Contains tools to evaluate the router's performance against static baselines.
    """
    
    @staticmethod
    def sequential_ab_test(queries: List[str], router: OptimizationRouter, baselines: List[str]) -> Dict[str, List[float]]:
        """
        Runs a sequential comparison of the contextual bandit router vs static baselines.
        
        Why not a naive single-shot t-test?
        1. Non-stationarity: Our traffic simulator introduces distribution shifts and model shocks. 
           Standard t-tests assume data is IID (Independent and Identically Distributed), which 
           is false here. 
        2. Peeking: In real-world streaming deployments, product managers look at metrics continuously. 
           Standard p-values become invalid if you stop the test as soon as it's significant. 
           We track cumulative reward over time to visualize exactly how the router adapts 
           when baselines fail during shifts.
        """
        logger.info(f"Running sequential A/B test over {len(queries)} queries.")
        
        cumulative_rewards = {"router": []}
        for b in baselines:
            cumulative_rewards[b] = []
            
        current_sums = {k: 0.0 for k in cumulative_rewards.keys()}
        
        for q in queries:
            # Evaluate router
            res_router = router.route_and_execute(q)
            current_sums["router"] += res_router["final_reward"]
            cumulative_rewards["router"].append(current_sums["router"])
            
            # Evaluate baselines (forcing the client and judge directly)
            for b in baselines:
                # Bypass bandit and force model selection
                resp_b = router.client.generate(b, q)
                score_b, _ = router.judge.evaluate(q, resp_b.response_text, b)
                cost_penalty = resp_b.simulated_cost * 0.1
                reward_b = max(0.0, score_b - cost_penalty)
                
                current_sums[b] += reward_b
                cumulative_rewards[b].append(current_sums[b])
                
        return cumulative_rewards

    @staticmethod
    def doubly_robust_off_policy_evaluation(
            logged_data: List[Dict], 
            target_policy: Callable[[Dict], str],
            reward_estimator: Callable[[Dict, str], float]) -> float:
        """
        Estimates the expected reward of a *new* target_policy using historical logs,
        without needing to run live traffic.
        
        Why not simple offline replay (Direct Method or pure IPS)?
        - Simple Offline Replay (Direct Method): We just predict what the reward would be 
          using a supervised model. This is biased if our reward model is wrong.
        - Inverse Propensity Scoring (IPS): We reweight logged rewards by (1/propensity). 
          This is unbiased but has extremely high variance because propensities can be tiny.
          
        Doubly Robust (DR) estimation combines both. It uses the reward_estimator as a baseline, 
        and uses IPS only on the *residual* (the error of the estimator). If either the 
        propensity or the reward model is accurate, the DR estimator is unbiased.
        """
        if not logged_data:
            return 0.0
            
        dr_scores = []
        
        for step_data in logged_data:
            # What the historical logging policy did
            logged_arm = step_data["model_used"]
            logged_reward = step_data["final_reward"]
            
            # In a real LinUCB, propensity isn't explicitly tracked as a probability 
            # like in Softmax exploration, but for the sake of the DR formula, we assume 
            # we logged the exploration probability (epsilon) or calculated it. 
            # We default to a small probability for fallback to avoid division by zero.
            propensity = step_data.get("propensity", 0.1) 
            
            # What the new policy would do
            target_arm = target_policy(step_data)
            
            # What our model predicts the reward would be
            predicted_reward_target = reward_estimator(step_data, target_arm)
            predicted_reward_logged = reward_estimator(step_data, logged_arm)
            
            # Doubly Robust Formula
            if target_arm == logged_arm:
                # If target policy agrees with log, we use actual reward + estimated counterfactual correction
                dr_estimate = predicted_reward_target + (logged_reward - predicted_reward_logged) / propensity
            else:
                # If target policy disagrees, we have to rely entirely on the reward estimator
                dr_estimate = predicted_reward_target
                
            dr_scores.append(dr_estimate)
            
        return float(np.mean(dr_scores))
