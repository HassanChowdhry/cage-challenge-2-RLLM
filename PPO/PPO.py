import torch
import torch.nn as nn

from .ActorCritic import ActorCritic, Memory


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO:
    """Basic Proximal Policy Optimization algorithm.

    This is a small adaptation of `PPO_discrete.py` from
    https://github.com/geekyutao/PyTorch-PPO .
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 2.5e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        gamma: float = 0.99,
        K_epochs: int = 4,
        eps_clip: float = 0.2,
    ) -> None:
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.mse = nn.MSELoss()

        self.memory = Memory()

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Sample an action using the old policy and record transition."""
        action = self.policy_old.act(state, self.memory)
        return action

    def _compute_returns(self) -> torch.Tensor:
        rewards = []
        discounted = 0.0
        for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if done:
                discounted = 0.0
            discounted = reward + self.gamma * discounted
            rewards.insert(0, discounted)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return rewards

    def update(self) -> None:
        returns = self._compute_returns()
        old_states = torch.cat(self.memory.states, 0).to(device)
        old_actions = torch.stack(self.memory.actions).to(device).squeeze()
        old_logprobs = torch.stack(self.memory.logprobs).to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = returns - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse(state_values, returns)
            entropy_loss = -dist_entropy.mean()
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()
