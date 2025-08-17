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
    agent1.save(filepath)

    # Create a new agent and load the state
    agent2 = PPOAgent(
        agent1.actor.network[0].in_features, agent1.actor.network[-2].out_features
    )
    agent2.load(filepath)

    # Compare the state dictionaries of the models
    assert agent1.actor.state_dict().keys() == agent2.actor.state_dict().keys()

    for p1, p2 in zip(agent1.actor.parameters(), agent2.actor.parameters()):
        assert torch.equal(p1, p2)

    for p1, p2 in zip(agent1.critic.parameters(), agent2.critic.parameters()):
        assert torch.equal(p1, p2)
