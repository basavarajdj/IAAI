"""
PPO Drift-Adaptation Agent
==========================

Architecture
------------
    drift signals + model context
              |
        [ drift encoder ]     2-layer MLP, shared trunk
              |
       +------+------+
       |             |
  [ policy head ] [ value head ]
   4 actions        scalar V(s)

The shared trunk is the "drift encoder" in the brief: it maps the raw detector
outputs into a representation the policy can act on. Sharing it between policy
and value is standard PPO practice and matters here because the dataset is tiny
— the value head's gradient is extra supervision for the encoder.

Why PPO
-------
The action space is discrete and small, episodes are short (14 steps), and the
environment is a cheap lookup table, so sample efficiency is not the binding
constraint — stability is. PPO's clipped objective prevents any single batch of
14-step episodes from moving the policy too far, which matters when the whole
dataset is one trajectory replayed with different action choices. Q-learning
variants would be more sample-efficient and much less stable at this size.

Exploration
-----------
* **Training:** the categorical policy is sampled from directly, with an entropy
  bonus to stop it collapsing onto "do nothing" early. Epsilon-greedy is also
  supported for the ablation the brief asks for, but sampling from the policy is
  what PPO is defined against.
* **Production:** `mode='thompson'` draws the action from the learned
  categorical policy instead of taking the argmax, so deployment keeps
  exploring in proportion to the agent's own uncertainty rather than at a fixed
  epsilon. If the network is built with `dropout > 0` this additionally becomes
  an MC-dropout draw over parameters.

  Naming honestly: with dropout off (the default, for the PPO-correctness
  reason in `DriftPolicyNet`) this is *posterior sampling over actions*, not
  full parameter-space Thompson sampling. It gives the practical property we
  want — uncertainty-proportional exploration with no tuned epsilon — but a
  true parameter posterior would need an ensemble or a Bayesian last layer.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl_env import N_ACTIONS, STATE_DIM

logger = logging.getLogger(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DriftPolicyNet(nn.Module):
    """Drift encoder + policy head + value head.

    Dropout defaults to 0. PPO's importance ratio compares the log-probability
    of an action under the current policy against its log-probability when the
    action was taken. Dropout makes the network stochastic, so if it is active
    during the update but not during rollout (or active independently in both),
    that ratio mixes policy change with dropout noise and the clipped objective
    silently stops meaning what it should. This was a real bug here: with
    dropout enabled the agent failed to learn a task whose optimum was obvious,
    converging to 'always retrain' and leaving 46% of the available return on
    the table.

    Set `dropout > 0` only if you intend MC-dropout Thompson sampling and accept
    the trade — see `PPOAgent.act`.
    """

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=128, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.encoder = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state):
        z = self.encoder(state)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    def distribution(self, state):
        logits, value = self(state)
        return Categorical(logits=logits), value


class PPOAgent:
    """Compact PPO: rollout, GAE, clipped update. No external RL dependency."""

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, lr=3e-4, gamma=0.95,
                 gae_lambda=0.95, clip_eps=0.2, entropy_coef=0.02, value_coef=0.5,
                 epochs_per_update=4, seed=42):
        torch.manual_seed(seed)
        self.net = DriftPolicyNet(state_dim, n_actions).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs_per_update = epochs_per_update
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

        # Normalisation statistics for the observation vector, learned online.
        # Drift signals live on wildly different scales (a PSI ratio of 12 next
        # to a KS fraction of 0.3); without this the encoder spends its capacity
        # undoing that instead of learning a policy.
        self._obs_mean = np.zeros(state_dim, dtype=np.float64)
        self._obs_var = np.ones(state_dim, dtype=np.float64)
        self._obs_count = 1e-4

    # ── observation normalisation ────────────────────────────────
    def _update_obs_stats(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        n = len(batch)
        if n == 0:
            return
        mean, var = batch.mean(axis=0), batch.var(axis=0)
        delta = mean - self._obs_mean
        total = self._obs_count + n
        self._obs_mean += delta * n / total
        self._obs_var = (self._obs_var * self._obs_count + var * n
                         + delta ** 2 * self._obs_count * n / total) / total
        self._obs_count = total

    def _normalise(self, obs):
        return np.clip((obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8), -10, 10)

    def _to_tensor(self, obs):
        return torch.as_tensor(self._normalise(obs), dtype=torch.float32, device=DEVICE)

    # ── acting ───────────────────────────────────────────────────
    def act(self, state, mode='sample', epsilon=0.0):
        """Choose an action.

        mode='sample'   — draw from the policy (PPO training)
        mode='greedy'   — argmax (deterministic evaluation)
        mode='thompson' — MC-dropout draw (production exploration)
        """
        if mode == 'sample' and epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions)), 0.0, 0.0

        # Thompson draws may enable dropout (MC-dropout posterior) if the net was
        # configured with it; everything else must stay deterministic so the PPO
        # ratio compares like with like.
        self.net.train(mode == 'thompson' and self._has_dropout())
        with torch.no_grad():
            dist, value = self.net.distribution(self._to_tensor(state).unsqueeze(0))
            action = torch.argmax(dist.probs, dim=-1) if mode == 'greedy' else dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def _has_dropout(self):
        return any(isinstance(m, nn.Dropout) for m in self.net.modules())

    def action_probabilities(self, state):
        self.net.eval()
        with torch.no_grad():
            dist, _ = self.net.distribution(self._to_tensor(state).unsqueeze(0))
        return dist.probs.cpu().numpy()[0]

    # ── rollout + update ─────────────────────────────────────────
    def collect_episode(self, env, epsilon=0.0):
        states, actions, log_probs, values, rewards = [], [], [], [], []
        state = env.reset()
        while True:
            action, log_prob, value = self.act(state, mode='sample', epsilon=epsilon)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            state = next_state
            if done:
                break
        return states, actions, log_probs, values, rewards

    def _advantages(self, rewards, values):
        """Generalised Advantage Estimation, bootstrapping from 0 at episode end."""
        adv, running = np.zeros(len(rewards), dtype=np.float32), 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * next_value - values[t]
            running = delta + self.gamma * self.gae_lambda * running
            adv[t] = running
        return adv, adv + np.asarray(values, dtype=np.float32)

    def update(self, batch):
        """One PPO update over a batch of episodes."""
        states, actions, old_log_probs, advantages, returns = [], [], [], [], []
        for s, a, lp, v, r in batch:
            adv, ret = self._advantages(r, v)
            states += s
            actions += a
            old_log_probs += lp
            advantages.append(adv)
            returns.append(ret)

        self._update_obs_stats(states)
        S = self._to_tensor(np.asarray(states))
        A = torch.as_tensor(actions, dtype=torch.long, device=DEVICE)
        OLD = torch.as_tensor(old_log_probs, dtype=torch.float32, device=DEVICE)
        ADV = torch.as_tensor(np.concatenate(advantages), dtype=torch.float32, device=DEVICE)
        RET = torch.as_tensor(np.concatenate(returns), dtype=torch.float32, device=DEVICE)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        stats = {}
        self.net.train()
        for _ in range(self.epochs_per_update):
            dist, value = self.net.distribution(S)
            log_probs = dist.log_prob(A)
            ratio = torch.exp(log_probs - OLD)

            unclipped = ratio * ADV
            clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * ADV
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = ((value - RET) ** 2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

            stats = {'policy_loss': float(policy_loss.item()),
                     'value_loss': float(value_loss.item()),
                     'entropy': float(entropy.item())}
        return stats

    # ── training loop ────────────────────────────────────────────
    def train(self, env, n_updates=150, episodes_per_update=8,
              epsilon_start=0.3, epsilon_end=0.02, log_every=25):
        """Train on repeated replays of the same weekly stream.

        Every episode replays identical data; only the agent's choices differ.
        That is exactly what the precomputed lattice buys — the environment is
        deterministic given the action sequence, so the agent explores the space
        of *policies* rather than needing more data.
        """
        history = []
        for update in range(n_updates):
            frac = update / max(n_updates - 1, 1)
            epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

            batch = [self.collect_episode(env, epsilon=epsilon)
                     for _ in range(episodes_per_update)]
            stats = self.update(batch)

            mean_return = float(np.mean([sum(ep[4]) for ep in batch]))
            history.append({'update': update, 'mean_return': mean_return,
                            'epsilon': epsilon, **stats})

            if update % log_every == 0 or update == n_updates - 1:
                greedy = env.run_policy(lambda s: self.act(s, mode='greedy')[0])
                logger.info(
                    f"  update {update:3d}  return {mean_return:7.3f}  "
                    f"entropy {stats['entropy']:.3f}  eps {epsilon:.2f}  | "
                    f"greedy: AUC {greedy['mean_auc']:.4f}, "
                    f"{greedy['n_full_retrains']}F/{greedy['n_partial_updates']}P/"
                    f"{greedy['n_hedges']}H"
                )
        return history

    # ── persistence ──────────────────────────────────────────────
    def save(self, path):
        torch.save({'net': self.net.state_dict(), 'obs_mean': self._obs_mean,
                    'obs_var': self._obs_var, 'obs_count': self._obs_count}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.net.load_state_dict(ckpt['net'])
        self._obs_mean, self._obs_var = ckpt['obs_mean'], ckpt['obs_var']
        self._obs_count = ckpt['obs_count']
        return self
