"""1次元 Flow Matching の最小実装.

本モジュールは、以下の処理を1ファイルで完結させるサンプル実装です。

1. 目標分布として1次元2峰性ガウス混合分布 (Mixture of Gaussians) を定義する。
2. Flow Matching の教師データ `(x_t, t, u)` を生成する。
3. 速度場 `v(x, t)` を MLP で学習する。
4. 学習前後のサンプル分布をヒストグラムで比較して可視化する。
"""

from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state

Array: TypeAlias = Any
Batch1D = tuple[Array, Array, Array]


def sample_x1_mog(key: Array, batch_size: int, mu: float = 2.0, sigma: float = 0.5) -> Array:
    """1次元2峰性ガウス混合分布からサンプルを生成する.

    ±`mu` を平均とする2つの正規分布を等確率で混合した分布から
    `batch_size` 個のサンプルを返す。

    Args:
        key: JAX の乱数キー。
        batch_size: 生成するサンプル数。
        mu: 各モードの中心の絶対値。
        sigma: 各モードの標準偏差。

    Returns:
        shape が `(batch_size, 1)` の1次元サンプル配列。
    """
    key_choice, key_noise = jax.random.split(key, 2)
    flag = jax.random.bernoulli(key_choice, p=0.5, shape=(batch_size, 1)).astype(jnp.float32)
    mean = (2.0 * flag - 1.0) * mu
    eps = jax.random.normal(key_noise, (batch_size, 1))
    x1 = mean + sigma * eps
    return x1


def make_fm_batch(key: Array, batch_size: int) -> Batch1D:
    """Flow Matching の教師データバッチを構築する.

    補間規則を `x_t = (1 - t) * x0 + t * x1` とし、教師信号を
    `u = x1 - x0` とする標準的な条件付き Flow Matching の形式で
    学習用バッチを返す。

    Args:
        key: JAX の乱数キー。
        batch_size: バッチサイズ。

    Returns:
        `(xt, t, u)` のタプル。
        `xt` は shape `(batch_size, 1)`、`t` は shape `(batch_size, 1)`、
        `u` は shape `(batch_size, 1)` を持つ。
    """
    k0, k1, kt = jax.random.split(key, 3)
    x0 = jax.random.normal(k0, (batch_size, 1))
    x1 = sample_x1_mog(k1, batch_size)
    t = jax.random.uniform(kt, (batch_size, 1))  # [0,1)
    xt = (1.0 - t) * x0 + t * x1
    u = x1 - x0
    return xt, t, u


class VelocityField(nn.Module):
    """1次元速度場 `v(x, t)` を近似する MLP.

    Attributes:
        hidden: 隠れ層のユニット数。
    """

    hidden: int = 128

    @nn.compact
    def __call__(self, x: Array, t: Array) -> Array:
        """入力 `(x, t)` から速度 `v` を推論する.

        Args:
            x: 現在位置。shape は `(B, 1)`。
            t: 時刻。shape は `(B, 1)`。

        Returns:
            推定速度。shape は `(B, 1)`。
        """
        h = jnp.concatenate([x, t], axis=-1)  # [B,2]
        h = nn.Dense(self.hidden)(h)
        h = nn.tanh(h)
        h = nn.Dense(self.hidden)(h)
        h = nn.tanh(h)
        v = nn.Dense(1)(h)  # [B,1]
        return v


def create_state(rng: Array, lr: float = 1e-3) -> tuple[train_state.TrainState, VelocityField]:
    """学習状態とモデルを初期化する.

    Args:
        rng: パラメータ初期化に使う乱数キー。
        lr: Adam の学習率。

    Returns:
        学習状態 `TrainState` とモデルインスタンスのタプル。
    """
    model = VelocityField(hidden=128)
    params = model.init(rng, jnp.zeros((1, 1)), jnp.zeros((1, 1)))["params"]
    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, model


@jax.jit
def train_step(
    state: train_state.TrainState, batch: Batch1D
) -> tuple[train_state.TrainState, Array]:
    """1ステップ分の学習更新を実行する.

    損失関数は平均二乗誤差 `E[||v(x_t,t)-u||^2]` を用いる。

    Args:
        state: 現在の学習状態。
        batch: 学習バッチ `(xt, t, u)`。

    Returns:
        更新後の `state` とスカラー損失。
    """
    xt, t, u = batch

    def loss_fn(params: Any) -> Array:
        v = state.apply_fn({"params": params}, xt, t)
        return jnp.mean(jnp.square(v - u))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def euler_sample(
    state: train_state.TrainState,
    key: Array,
    n_samples: int,
    n_steps: int = 50,
) -> Array:
    """Euler 法で常微分方程式 `dx/dt = v(x,t)` を積分してサンプリングする.

    Args:
        state: 学習済みまたは未学習の速度場を含む学習状態。
        key: 初期点 `x0 ~ N(0,1)` の生成に使う乱数キー。
        n_samples: サンプル数。
        n_steps: 時間離散化ステップ数。大きいほど積分誤差が小さくなる。

    Returns:
        近似された終端サンプル `x1`。shape は `(n_samples, 1)`。
    """
    x = jax.random.normal(key, (n_samples, 1))
    dt = 1.0 / n_steps

    def step_fn(x_curr: Array, k: Array) -> tuple[Array, None]:
        t = jnp.full((n_samples, 1), k * dt)
        v = state.apply_fn({"params": state.params}, x_curr, t)
        x_next = x_curr + dt * v
        return x_next, None

    x, _ = jax.lax.scan(step_fn, x, jnp.arange(n_steps))
    return x


def main(
    seed: int = 0,
    batch_size: int = 2048,
    steps: int = 5000,
    lr: float = 1e-3,
    n_samples: int = 20000,
    n_steps: int = 60,
) -> tuple[Array, Array, Array]:
    """Flow Matching の学習と評価用サンプリングを実行する.

    学習前サンプル・学習後サンプル・目標分布サンプルを返す。

    Args:
        seed: 乱数シード。
        batch_size: 学習バッチサイズ。
        steps: 学習反復回数。
        lr: 学習率。
        n_samples: 可視化用サンプル数。
        n_steps: Euler サンプリングの積分ステップ数。

    Returns:
        `(pre_train_samples, post_train_samples, target_samples)`。
        いずれも shape は `(n_samples, 1)`。
    """
    key = jax.random.PRNGKey(seed)
    key_init, key_train, key_pre_sample, key_post_sample, key_target = jax.random.split(key, 5)

    state, model = create_state(key_init, lr=lr)
    del model

    pre_train_samples = euler_sample(state, key_pre_sample, n_samples=n_samples, n_steps=n_steps)

    k = key_train
    for i in range(steps):
        k, kb = jax.random.split(k)
        batch = make_fm_batch(kb, batch_size)
        # state, loss = train_step(state, model, batch)
        state, loss = train_step(state, batch)

        if (i + 1) % 500 == 0:
            print(f"step {i + 1:5d} | loss {float(loss):.4f}")

    # xs = euler_sample(state.params, model, key_post_sample, n_samples=n_samples, n_steps=n_steps)
    post_train_samples = euler_sample(state, key_post_sample, n_samples=n_samples, n_steps=n_steps)
    target_samples = sample_x1_mog(key_target, n_samples)

    return pre_train_samples, post_train_samples, target_samples


if __name__ == "__main__":
    """スクリプト実行時のエントリポイント."""
    pre_xs, post_xs, target_xs = main()

    pre_xs_np = np.asarray(pre_xs).reshape(-1)
    post_xs_np = np.asarray(post_xs).reshape(-1)
    target_xs_np = np.asarray(target_xs).reshape(-1)

    plt.figure(figsize=(10, 6))
    plt.hist(
        target_xs_np,
        bins=120,
        density=True,
        alpha=0.25,
        label="target data (x1, MoG)",
        color="gray",
    )
    plt.hist(
        pre_xs_np,
        bins=120,
        density=True,
        alpha=0.45,
        label="before training",
        color="tab:orange",
    )
    plt.hist(
        post_xs_np,
        bins=120,
        density=True,
        alpha=0.45,
        label="after training",
        color="tab:blue",
    )
    plt.title("Flow Matching sample distribution: before vs after training")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()
