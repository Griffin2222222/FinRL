import numpy as np


class GPTContextualReward:
    """
    Class to adjust rewards based on macroeconomic context using a GPT client or rule-based logic.
    """

    def __init__(self, gpt_client=None):
        """
        Args:
            gpt_client: callable or object, must have .get_reward_adjustment(context, action)
        """
        self.gpt_client = gpt_client

    def __call__(self, base_reward, macro_context, action):
        """
        Adjust reward based on macroeconomic news/context using GPT or rules.
        Args:
            base_reward: float, the original reward
            macro_context: str, macro news or summary
            action: np.ndarray or float, the agent's action
        Returns:
            float: adjusted reward
        """
        if self.gpt_client is not None:
            adjustment = self.gpt_client.get_reward_adjustment(macro_context, action)
        else:
            # Fallback: simple rule-based adjustment
            context = macro_context.lower()
            if "recession" in context:
                adjustment = (
                    -abs(base_reward) * 0.5
                )  # penalize risky actions in recession
            elif "boom" in context:
                adjustment = abs(base_reward) * 0.5  # reward risk-taking in boom
            else:
                adjustment = 0
        return base_reward + adjustment


class DummyGPTClient:
    """
    Example GPT client stub for reward adjustment.
    """

    def get_reward_adjustment(self, context, action):
        context = context.lower()
        # Ensure action is a numpy array for consistent behavior
        if isinstance(action, (float, int)):
            action_sum = abs(action)
        else:
            action_sum = np.abs(action).sum()
        if "hawkish" in context:
            return -0.1 * action_sum
        elif "dovish" in context:
            return 0.1 * action_sum
        return 0


# Example usage:
if __name__ == "__main__":
    base_reward = 1.0
    macro_context = "The central bank issued a hawkish statement."
    action = np.array([0.5, -0.2, 0.3])

    # Using DummyGPTClient
    gpt_client = DummyGPTClient()
    contextual_reward = GPTContextualReward(gpt_client=gpt_client)
    adjusted_reward = contextual_reward(base_reward, macro_context, action)
    print(f"Adjusted reward (with GPT client): {adjusted_reward}")

    # Using rule-based fallback
    contextual_reward_rule = GPTContextualReward()
    adjusted_reward_rule = contextual_reward_rule(
        base_reward, "Economy is in recession.", action
    )
    print(f"Adjusted reward (rule-based): {adjusted_reward_rule}")
