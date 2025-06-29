import numpy as np
from typing import Union


def blend_actions(
    rl_action: Union[np.ndarray, float],
    sentiment_score: Union[np.ndarray, float],
    sentiment_confidence: Union[np.ndarray, float],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend RL action and sentiment signal using confidence-weighted logic.

    Args:
        rl_action: np.ndarray or float, action(s) from RL agent (e.g., [-1, 0, 1])
        sentiment_score: np.ndarray or float, sentiment signal(s) from FinGPT (e.g., [-1, 1])
        sentiment_confidence: np.ndarray or float, confidence in sentiment (0-1)
        alpha: float, base blending factor (0=only RL, 1=only sentiment)

    Returns:
        np.ndarray: blended action(s)
    """
    rl_action = np.asarray(rl_action)
    sentiment_score = np.asarray(sentiment_score)
    sentiment_confidence = np.asarray(sentiment_confidence)

    # Weighted blend: higher confidence in sentiment increases its influence
    weight = alpha * sentiment_confidence
    blended = (1 - weight) * rl_action + weight * sentiment_score
    return np.clip(blended, -1, 1)


# Example usage
if __name__ == "__main__":
    rl_action = np.array([0.5, -0.2, 1.0])
    sentiment_score = np.array([1.0, -1.0, 0.0])
    sentiment_confidence = np.array([0.8, 0.3, 0.5])
    blended = blend_actions(rl_action, sentiment_score, sentiment_confidence, alpha=0.7)
    print("Blended actions:", blended)
