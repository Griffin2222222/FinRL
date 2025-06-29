import numpy as np


class DummyAgent:
    """
    A simple agent for demonstration. Replace with your actual agent implementation.
    """

    def __init__(self, name):
        self.name = name

    def act(self, state):
        # Dummy action: just return the agent's name and state for demonstration
        return {"agent": self.name, "action": "hold", "state": state}


class MetaAgent:
    def __init__(self, agents, regime_detector):
        """
        Args:
            agents: dict, e.g. {'bull': agent1, 'bear': agent2, 'sideways': agent3}
            regime_detector: callable, takes state/history and returns 'bull', 'bear', or 'sideways'
        """
        self.agents = agents
        self.regime_detector = regime_detector

    def select_agent(self, state, history=None):
        regime = self.regime_detector(state, history)
        return self.agents.get(regime, self.agents["sideways"])

    def act(self, state, history=None):
        agent = self.select_agent(state, history)
        return agent.act(state)


def simple_regime_detector(state, history=None, window=20):
    """
    Detects market regime based on rolling returns.
    Returns: 'bull', 'bear', or 'sideways'
    """
    if history is None or len(history) < window:
        return "sideways"
    returns = np.diff(history[-window:]) / np.array(history[-window:-1])
    mean_ret = np.mean(returns)
    if mean_ret > 0.002:
        return "bull"
    elif mean_ret < -0.002:
        return "bear"
    else:
        return "sideways"


# Example usage
if __name__ == "__main__":
    # Create dummy agents for each regime
    bull_agent = DummyAgent("bull")
    bear_agent = DummyAgent("bear")
    sideways_agent = DummyAgent("sideways")

    agents = {"bull": bull_agent, "bear": bear_agent, "sideways": sideways_agent}

    meta_agent = MetaAgent(agents, simple_regime_detector)

    # Simulate some price history
    price_history = np.linspace(100, 120, 25)  # Simulate a bull market
    state = price_history[-1]

    action = meta_agent.act(state, price_history)
    print("Action in bull market:", action)

    # Simulate a bear market
    price_history = np.linspace(120, 100, 25)
    state = price_history[-1]
    action = meta_agent.act(state, price_history)
    print("Action in bear market:", action)

    # Simulate a sideways market
    price_history = np.ones(25) * 110 + np.random.normal(0, 0.1, 25)
    state = price_history[-1]
    action = meta_agent.act(state, price_history)
    print("Action in sideways market:", action)
