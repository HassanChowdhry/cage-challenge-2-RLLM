import pytest

torch = pytest.importorskip("torch")
from PPO.ActorCritic import ActorCritic


def test_act():
    model = ActorCritic(state_dim=4, action_dim=2)
    state = torch.zeros(1, 4)
    action = model.act(state)
    assert action.item() in {0, 1}

def test_ppo_update():
    from PPO import PPO
    algo = PPO(state_dim=4, action_dim=2)
    state = torch.zeros(1, 4)
    action = algo.select_action(state)
    algo.memory.rewards.append(1.0)
    algo.memory.is_terminals.append(True)
    algo.update()
    assert len(algo.memory.states) == 0
