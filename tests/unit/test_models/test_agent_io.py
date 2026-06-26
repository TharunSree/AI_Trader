# tests/unit/test_models/test_agent_io.py

import pytest
import torch
from pathlib import Path
from src.models.ppo_agent import PPOAgent


@pytest.fixture
def agent_and_path(tmp_path: Path):
    """Fixture to create an agent and a temporary file path."""
    state_dim = 10
    action_dim = 3
    agent = PPOAgent(state_dim, action_dim)
    filepath = tmp_path / "test_agent.pth"
    return agent, filepath


def test_save_load_agent(agent_and_path):
    """Tests if an agent's state can be saved and loaded correctly."""
    agent1, filepath = agent_and_path

    # Save the initial state
    agent1.save_weights(filepath)

    # Create a new agent and load the state
    agent2 = PPOAgent(10, 3)
    agent2.load_weights(filepath)

    # Compare the state dictionaries of the models
    assert agent1.policy.actor.state_dict().keys() == agent2.policy.actor.state_dict().keys()

    for p1, p2 in zip(agent1.policy.actor.parameters(), agent2.policy.actor.parameters()):
        assert torch.equal(p1, p2)

    for p1, p2 in zip(agent1.policy.critic.parameters(), agent2.policy.critic.parameters()):
        assert torch.equal(p1, p2)

