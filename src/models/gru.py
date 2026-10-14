"""A minimal multi-task GRU in NumPy — the learned sequence model (reviewer #4).

Why NumPy, not PyTorch
----------------------
The project is CPU-only and ``tests/test_baselines.py::test_cpu_only_execution_path``
asserts that the baseline/metrics stack imports no torch/CUDA. The CPU-only
PyTorch wheel lives on ``download.pytorch.org``, which is outside this
environment's allowed hosts, so a NumPy GRU is the honest way to ship a learned
sequence model here. It is still a genuine GRU: gated recurrent updates, trained
end-to-end by backpropagation through time (verified by a numerical gradient
check in ``tests/test_gru.py``), with two heads exactly as the spec requires —
``G`` (episode-level) and ``I_t`` (per step).

Contract
--------
The input at position ``t`` is ``frame_features(O_t)`` from
``src/features/timing.py`` (the same contract the probes use). The goal head is
read off the hidden state at the ``α``-truncated index, so the learned model and
the closed-form baselines are compared on identical prefixes
(``TIMING_CONTRACT.md`` §6). Any interaction-masked arm must use noise
substitution, not zeroing (``TIMING_CONTRACT.md`` §8).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _xavier(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


class GRUParams:
    """Parameters of a single GRU layer plus the two task heads."""

    def __init__(self, input_dim: int, hidden: int, n_goals: int, n_intentions: int,
                 seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        h = hidden
        d = input_dim
        # gates: reset r, update z, candidate n; input weights (d,h) and
        # recurrent weights (h,h) and biases (h,)
        self.Wr, self.Ur, self.br = _xavier(d, h, rng), _xavier(h, h, rng), np.zeros(h)
        self.Wz, self.Uz, self.bz = _xavier(d, h, rng), _xavier(h, h, rng), np.zeros(h)
        self.Wh, self.Uh, self.bh = _xavier(d, h, rng), _xavier(h, h, rng), np.zeros(h)
        self.bz[:] = 0.0
        # heads
        self.Wg, self.bg = _xavier(h, n_goals, rng), np.zeros(n_goals)
        self.Wi, self.bi = _xavier(h, n_intentions, rng), np.zeros(n_intentions)

    def names(self) -> List[str]:
        return [n for n in self.__dict__ if isinstance(getattr(self, n), np.ndarray)]

    def zero_grad(self) -> Dict[str, np.ndarray]:
        return {n: np.zeros_like(getattr(self, n)) for n in self.names()}


class MultiTaskGRU:
    """GRU over observation frames with a goal head and an intention head."""

    def __init__(self, input_dim: int, hidden: int = 32, n_goals: int = 4,
                 n_intentions: int = 4, seed: int = 0) -> None:
        self.p = GRUParams(input_dim, hidden, n_goals, n_intentions, seed)
        self.hidden = hidden
        self.input_dim = input_dim
        self.n_goals = n_goals
        self.n_intentions = n_intentions

    # ------------------------------------------------------------ forward
    def _step(self, x: np.ndarray, h: np.ndarray) -> Tuple[dict, np.ndarray]:
        cache = {"x": x, "h_prev": h}
        r = sigmoid(x @ self.p.Wr + h @ self.p.Ur + self.p.br)
        z = sigmoid(x @ self.p.Wz + h @ self.p.Uz + self.p.bz)
        n = np.tanh(x @ self.p.Wh + (r * h) @ self.p.Uh + self.p.bh)
        h_new = (1.0 - z) * h + z * n
        cache.update(r=r, z=z, n=n, h=h_new)
        return cache, h_new

    def forward(self, X: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """X: (T, D). Returns per-step caches and the hidden states (T, H)."""
        h = np.zeros(self.hidden)
        caches = []
        hs = []
        for t in range(X.shape[0]):
            cache, h = self._step(X[t], h)
            caches.append(cache)
            hs.append(h)
        return caches, np.asarray(hs)

    def goal_logits(self, h: np.ndarray) -> np.ndarray:
        return h @ self.p.Wg + self.p.bg

    def intention_logits(self, hs: np.ndarray) -> np.ndarray:
        return hs @ self.p.Wi + self.p.bi

    # ------------------------------------------------------------ backward
    def backward(
        self,
        caches: List[dict],
        X: np.ndarray,
        goal_label: int,
        intention_labels: np.ndarray,
        alpha_index: int,
        intention_weight: float = 1.0,
    ) -> Tuple[Dict[str, np.ndarray], float, np.ndarray, np.ndarray]:
        """BPTT for one episode. Returns grads, loss, goal-probs, intention-probs."""
        p = self.p
        T = len(caches)
        H = self.hidden
        g = self.zero_grad()
        loss = 0.0

        hs = np.asarray([c["h"] for c in caches])
        gi = self.goal_logits(hs[alpha_index])
        gp = softmax(gi)
        loss -= np.log(max(gp[goal_label], 1e-12))
        # intention probs over all steps
        ii = self.intention_logits(hs)
        ip = softmax(ii, axis=1)
        for t in range(T):
            loss -= intention_weight * np.log(max(ip[t, intention_labels[t]], 1e-12))

        # gradient w.r.t. hidden states, accumulated over time
        dh = np.zeros((T, H))
        # goal head
        dlogits = gp.copy()
        dlogits[goal_label] -= 1.0
        g["Wg"] += np.outer(hs[alpha_index], dlogits)
        g["bg"] += dlogits
        dh[alpha_index] += dlogits @ p.Wg.T
        # intention head
        dip = ip.copy()
        for t in range(T):
            dip[t, intention_labels[t]] -= 1.0
        dip *= intention_weight
        g["Wi"] += hs.T @ dip
        g["bi"] += dip.sum(axis=0)
        dh += dip @ p.Wi.T

        # backprop through time
        for t in reversed(range(T)):
            c = caches[t]
            h_prev = c["h_prev"]
            r, z, n = c["r"], c["z"], c["n"]
            x = c["x"]
            dhn = dh[t]
            # h_new = (1-z)*h_prev + z*n
            dz = (n - h_prev) * dhn
            dn = z * dhn
            dh_prev = (1.0 - z) * dhn
            # n = tanh(...)
            dn_pre = (1.0 - n * n) * dn
            g["Wh"] += np.outer(x, dn_pre)
            g["bh"] += dn_pre
            d_rh = dn_pre @ p.Uh.T
            g["Uh"] += np.outer(r * h_prev, dn_pre)
            # the candidate also reads h through (r*h): d/dh of that term is r*d_rh
            dh_prev += d_rh * r
            dr = d_rh * h_prev
            # r = sigmoid(...)
            dr_pre = dr * r * (1.0 - r)
            g["Wr"] += np.outer(x, dr_pre)
            g["br"] += dr_pre
            dh_prev += dr_pre @ p.Ur.T
            # z = sigmoid(...)
            dz_pre = dz * z * (1.0 - z)
            g["Wz"] += np.outer(x, dz_pre)
            g["bz"] += dz_pre
            dh_prev += dz_pre @ p.Uz.T
            if t > 0:
                dh[t - 1] += dh_prev
        return g, float(loss), gp, ip

    def zero_grad(self) -> Dict[str, np.ndarray]:
        return self.p.zero_grad()


class Adam:
    def __init__(self, params: GRUParams, lr: float = 1e-2, b1: float = 0.9,
                 b2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = {n: np.zeros_like(v) for n, v in params.__dict__.items()
                  if isinstance(v, np.ndarray)}
        self.v = {n: np.zeros_like(v) for n, v in params.__dict__.items()
                  if isinstance(v, np.ndarray)}
        self.t = 0

    def step(self, params: GRUParams, grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for n in self.m:
            grad = grads[n]
            self.m[n] = self.b1 * self.m[n] + (1 - self.b1) * grad
            self.v[n] = self.b2 * self.v[n] + (1 - self.b2) * grad * grad
            mhat = self.m[n] / (1 - self.b1 ** self.t)
            vhat = self.v[n] / (1 - self.b2 ** self.t)
            getattr(params, n)[...] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def numerical_gradient_check(model: "MultiTaskGRU", X: np.ndarray, goal: int,
                             intentions: np.ndarray, alpha_index: int,
                             eps: float = 1e-5, seed: int = 0) -> float:
    """Max relative error between analytic BPTT and finite differences."""
    rng = np.random.default_rng(seed)
    names = [n for n in model.p.names() if n in ("Wr", "Wz", "Wh", "Wg", "Wi", "br")]
    grads, _, _, _ = _loss_and_grads(model, X, goal, intentions, alpha_index)
    worst = 0.0
    for n in names:
        param = getattr(model.p, n)
        flat = param.reshape(-1)
        for _ in range(3):
            idx = int(rng.integers(flat.size))
            orig = flat[idx]
            flat[idx] = orig + eps
            lp = _forward_loss(model, X, goal, intentions, alpha_index)
            flat[idx] = orig - eps
            lm = _forward_loss(model, X, goal, intentions, alpha_index)
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[n].reshape(-1)[idx]
            denom = max(abs(num), abs(ana), 1e-8)
            worst = max(worst, abs(num - ana) / denom)
    return worst


def _forward_loss(model, X, goal, intentions, alpha_index) -> float:
    _, hs = model.forward(X)
    gp = softmax(model.goal_logits(hs[alpha_index]))
    loss = -np.log(max(gp[goal], 1e-12))
    ip = softmax(model.intention_logits(hs), axis=1)
    for t in range(len(intentions)):
        loss -= np.log(max(ip[t, intentions[t]], 1e-12))
    return float(loss)


def _loss_and_grads(model, X, goal, intentions, alpha_index):
    caches, _ = model.forward(X)
    return model.backward(caches, X, goal, intentions, alpha_index)
