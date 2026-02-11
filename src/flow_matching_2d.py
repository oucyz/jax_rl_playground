"""2次元 Flow Matching のサンプル実装.

本モジュールは、2次元ガウス混合分布を目標分布として Flow Matching を学習し、
学習前・学習途中・学習後の生成サンプルを比較可視化する。
"""

from collections.abc import Sequence
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from matplotlib.lines import Line2D
from numpy.typing import NDArray

Array: TypeAlias = Any
NpArray: TypeAlias = NDArray[np.float64]
Batch2D = tuple[Array, Array, Array]
SnapshotMap = dict[int, Array]


def sample_x1_mog_2d(
    key: Array,
    batch_size: int,
    radius: float = 2.0,
    sigma: float = 0.35,
) -> Array:
    """2次元ガウス混合分布からサンプルを生成する.

    `(+r,+r), (+r,-r), (-r,+r), (-r,-r)` の4中心を持つ等確率混合分布を使う。

    Args:
        key: JAX の乱数キー。
        batch_size: 生成するサンプル数。
        radius: 各クラスタ中心の原点からの距離パラメータ。
        sigma: 各クラスタの標準偏差。

    Returns:
        shape が `(batch_size, 2)` の2次元サンプル配列。
    """
    key_choice, key_noise = jax.random.split(key, 2)
    centers = jnp.array(
        [
            [radius, radius],
            [radius, -radius],
            [-radius, radius],
            [-radius, -radius],
        ],
        dtype=jnp.float32,
    )
    idx = jax.random.randint(key_choice, (batch_size,), minval=0, maxval=centers.shape[0])
    means = centers[idx]  # [B,2]
    eps = jax.random.normal(key_noise, (batch_size, 2))
    x1 = means + sigma * eps
    return x1


def make_fm_batch(key: Array, batch_size: int) -> Batch2D:
    """Flow Matching 学習用の2次元バッチを生成する.

    Args:
        key: JAX の乱数キー。
        batch_size: バッチサイズ。

    Returns:
        `(xt, t, u)` のタプル。
        `xt` は shape `(batch_size, 2)`、`t` は shape `(batch_size, 1)`、
        `u` は shape `(batch_size, 2)` を持つ。
    """
    k0, k1, kt = jax.random.split(key, 3)
    x0 = jax.random.normal(k0, (batch_size, 2))
    x1 = sample_x1_mog_2d(k1, batch_size)
    t = jax.random.uniform(kt, (batch_size, 1))  # [0,1)
    xt = (1.0 - t) * x0 + t * x1
    u = x1 - x0
    return xt, t, u


class VelocityField(nn.Module):
    """2次元速度場 `v(x, t)` を近似する MLP.

    Attributes:
        hidden: 隠れ層のユニット数。
    """

    hidden: int = 128

    @nn.compact
    def __call__(self, x: Array, t: Array) -> Array:
        """2次元位置と時刻から速度を推定する.

        Args:
            x: 位置ベクトル。shape は `(B, 2)`。
            t: 時刻。shape は `(B, 1)`。

        Returns:
            推定速度。shape は `(B, 2)`。
        """
        h = jnp.concatenate([x, t], axis=-1)  # [B,3]
        h = nn.Dense(self.hidden)(h)
        h = nn.tanh(h)
        h = nn.Dense(self.hidden)(h)
        h = nn.tanh(h)
        v = nn.Dense(2)(h)  # [B,2]
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
    params = model.init(rng, jnp.zeros((1, 2)), jnp.zeros((1, 1)))["params"]
    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, model


@jax.jit
def train_step(
    state: train_state.TrainState, batch: Batch2D
) -> tuple[train_state.TrainState, Array]:
    """1ステップ分のパラメータ更新を行う.

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
    """Euler 法で 2D ODE `dx/dt = v(x,t)` を積分してサンプル生成する.

    Args:
        state: 学習状態。
        key: 初期点生成に使う乱数キー。
        n_samples: サンプル数。
        n_steps: 積分ステップ数。

    Returns:
        終端時刻のサンプル。shape は `(n_samples, 2)`。
    """
    x = jax.random.normal(key, (n_samples, 2))
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
    steps: int = 3000,
    lr: float = 1e-3,
    n_samples: int = 20000,
    n_steps: int = 60,
    snapshot_steps: Sequence[int] = (500, 1000, 2000),
) -> tuple[Array, SnapshotMap, Array, Array]:
    """Flow Matching 学習を実行し、可視化用サンプルを収集する.

    Args:
        seed: 乱数シード。
        batch_size: 学習バッチサイズ。
        steps: 学習反復回数。
        lr: 学習率。
        n_samples: 可視化に使うサンプル数。
        n_steps: Euler サンプリングの時間分割数。
        snapshot_steps: 学習途中サンプルを保存するステップ番号の列。

    Returns:
        以下を含むタプル。
        - `pre_train_samples`: 学習前サンプル。shape `(n_samples, 2)`。
        - `snapshots`: キーがステップ番号、値がその時点のサンプル配列の辞書。
        - `post_train_samples`: 学習後サンプル。shape `(n_samples, 2)`。
        - `target_samples`: 目標分布サンプル。shape `(n_samples, 2)`。
    """
    key = jax.random.PRNGKey(seed)
    key_init, key_train, key_pre_sample, key_target, key_loop = jax.random.split(key, 5)

    state, model = create_state(key_init, lr=lr)
    del model

    pre_train_samples = euler_sample(state, key_pre_sample, n_samples=n_samples, n_steps=n_steps)
    target_samples = sample_x1_mog_2d(key_target, n_samples)

    snapshot_steps = tuple(sorted({int(s) for s in snapshot_steps if 0 < int(s) < steps}))
    snapshots: SnapshotMap = {}
    sample_key = key_loop

    k = key_train
    for i in range(steps):
        k, kb = jax.random.split(k)
        batch = make_fm_batch(kb, batch_size)
        # state, loss = train_step(state, model, batch)
        state, loss = train_step(state, batch)

        if (i + 1) % 500 == 0:
            print(f"step {i + 1:5d} | loss {float(loss):.4f}")

        step_num = i + 1
        if step_num in snapshot_steps:
            sample_key, ks = jax.random.split(sample_key)
            snapshots[step_num] = euler_sample(state, ks, n_samples=n_samples, n_steps=n_steps)

    sample_key, key_post_sample = jax.random.split(sample_key)
    post_train_samples = euler_sample(state, key_post_sample, n_samples=n_samples, n_steps=n_steps)

    return pre_train_samples, snapshots, post_train_samples, target_samples


def _plot_samples_grid(
    pre_samples: Array,
    snapshots: SnapshotMap,
    post_samples: Array,
    target_samples: Array,
    max_points: int | None = 5000,
    contour_bins: int = 100,
    contour_levels: int = 7,
    target_color: str = "#9aa0a6",
    point_alpha: float = 0.50,
    point_size: float = 5.0,
) -> None:
    """学習各時点サンプルを目標分布等高線と重ねてグリッド可視化する.

    Args:
        pre_samples: 学習前サンプル。shape `(N, 2)`。
        snapshots: 学習途中スナップショット辞書。
        post_samples: 学習後サンプル。shape `(N, 2)`。
        target_samples: 目標分布サンプル。shape `(N, 2)`。
        max_points: 各パネルで描画する最大点数。`None` なら全点描画する。
        contour_bins: target 密度推定の2次元ヒストグラム分割数。
        contour_levels: target 等高線の本数。
        target_color: target 等高線の描画色。
        point_alpha: 各時点サンプル点の透過率。
        point_size: 各時点サンプル点のサイズ。
    """

    def _subsample_evenly(arr: np.ndarray, limit: int | None) -> np.ndarray:
        if limit is None or arr.shape[0] <= limit:
            return arr
        idx = np.linspace(0, arr.shape[0] - 1, limit, dtype=np.int32)
        return arr[idx]

    plt.style.use("seaborn-v0_8-whitegrid")

    target_arr = np.asarray(target_samples)
    entries = [("before (step 0)", np.asarray(pre_samples), "tab:orange")]
    entries += [
        (f"step {s}", np.asarray(snapshots[s]), "tab:blue") for s in sorted(snapshots.keys())
    ]
    entries.append(("after (final)", np.asarray(post_samples), "tab:green"))

    all_arrays = [target_arr] + [arr for _, arr, _ in entries]
    combined = np.concatenate(all_arrays, axis=0)
    x_min, y_min = np.min(combined, axis=0)
    x_max, y_max = np.max(combined, axis=0)
    x_pad = max(0.2, 0.05 * float(x_max - x_min))
    y_pad = max(0.2, 0.05 * float(y_max - y_min))
    x_lim = (float(x_min - x_pad), float(x_max + x_pad))
    y_lim = (float(y_min - y_pad), float(y_max + y_pad))

    density, x_edges, y_edges = np.histogram2d(
        target_arr[:, 0],
        target_arr[:, 1],
        bins=contour_bins,
        range=[x_lim, y_lim],
        density=True,
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx: NpArray
    yy: NpArray
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    contour_z = density.T

    total = len(entries)
    ncols = min(3, total)
    nrows = int(np.ceil(total / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.reshape(-1)

    for i, (title, samples, color) in enumerate(entries):
        arr = _subsample_evenly(samples, max_points)
        ax = axes_flat[i]
        ax.contour(
            xx,
            yy,
            contour_z,
            levels=contour_levels,
            colors=target_color,
            linewidths=1,
            alpha=0.85,
        )
        ax.scatter(arr[:, 0], arr[:, 1], s=point_size, alpha=point_alpha, color=color)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

    for j in range(total, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Flow Matching 2D: snapshot vs target density", fontsize=15, y=0.985)
    legend_handles = [
        Line2D([0], [0], color=target_color, lw=1.4, label="target density"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:blue",
            markeredgecolor="none",
            alpha=point_alpha,
            markersize=6,
            label="generated samples",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.92))
    plt.show()


if __name__ == "__main__":
    """スクリプト実行時のエントリポイント."""
    pre_xs, snapshot_xs, post_xs, target_xs = main(
        snapshot_steps=(200, 500, 1000),
        # snapshot_steps=(200, 500, 1000, 2000, 4000),
    )
    _plot_samples_grid(pre_xs, snapshot_xs, post_xs, target_xs)
