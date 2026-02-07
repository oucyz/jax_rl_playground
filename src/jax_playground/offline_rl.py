"""JAX で実装した学習用オフラインRLサンプル。

このモジュールは、オフラインRLの最小構成を学ぶための教材です。
実運用向けの高性能実装ではなく、アルゴリズムの流れを追いやすいことを
優先しています。

主な流れは次の 3 ステップです。
1. ふるまい方策から固定データセットを作る。
2. その静的データのみを使って Q 関数を学習する。
3. CQL 風の保守的項で、未観測行動の過大評価を抑える。

Example:
    REPL やスクリプトから最短で試す例:

    >>> from jax_playground.offline_rl import run_demo
    >>> result = run_demo(seed=0)
    >>> sorted(result.keys())
    ['action_left_state', 'action_right_state', 'first_loss', 'last_loss']

    モジュールを直接実行する例:

    $ PYTHONPATH=src python -m jax_playground.offline_rl
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import optax

Array: TypeAlias = Any


@dataclass(frozen=True)
class OfflineBatch:
    """オフラインRLで利用する固定遷移データを表すコンテナ。

    Attributes:
        observations: 現在状態。形状は ``[N, obs_dim]``。
            N は遷移サンプル数で、各行が 1 ステップ分の状態ベクトル。
        actions: 実行行動。形状は ``[N]`` の整数配列。
            各要素は離散行動ID（このサンプルでは 0 または 1）。
        rewards: 即時報酬。形状は ``[N]``。
            各遷移で観測された 1 ステップ報酬。
        next_observations: 次状態。形状は ``[N, obs_dim]``。
            ``observations`` と同じ順序で対応する次時刻の状態。
        dones: 終端フラグ。形状は ``[N]``。
            1.0 は終端遷移、0.0 は非終端遷移を意味する。
    """

    observations: Array
    actions: Array
    rewards: Array
    next_observations: Array
    dones: Array


@dataclass(frozen=True)
class TrainConfig:
    """保守的オフラインQ学習で使う設定値をまとめたクラス。

    Attributes:
        obs_dim: 状態ベクトルの次元数。
            このサンプル環境は 1 次元位置なので既定値は 1。
        n_actions: 離散行動数。
            このサンプルは「左移動 / 右移動」の 2 行動を想定。
        hidden_dim: Q ネットワーク隠れ層のユニット数。
            大きいほど表現力は上がるが、学習コストも増える。
        learning_rate: Adam オプティマイザの学習率。
            値が大きすぎると発散しやすく、小さすぎると収束が遅い。
        gamma: 割引率（0.0 〜 1.0）。
            将来報酬をどれだけ重視するかを決める係数。
            1.0 に近いほど長期的な報酬を重視する。
        cql_alpha: 保守的項の重み係数。
            大きいほど未観測行動の Q 値を抑える圧力が強くなる。
        tau: ターゲットネットワーク更新の Polyak 係数。
            ``target <- (1 - tau) * target + tau * online`` で更新する。
            小さいほどターゲットがゆっくり追従して学習が安定しやすい。
        batch_size: 1 ステップ学習で使うミニバッチサイズ。
            大きいほど勾配分散は小さくなるが計算負荷は増える。
        train_steps: 学習ループの反復回数。
            学習の進み具合と計算時間をトレードオフで調整する。
    """

    obs_dim: int = 1
    n_actions: int = 2
    hidden_dim: int = 32
    learning_rate: float = 3e-3
    gamma: float = 0.98
    cql_alpha: float = 1.0
    tau: float = 0.01
    batch_size: int = 128
    train_steps: int = 500


@dataclass(frozen=True)
class TrainMetrics:
    """学習中の損失指標を保持するクラス。

    Attributes:
        total_loss: 最適化対象の総損失。
            ``bellman_loss + cql_alpha * conservative_loss`` で定義される。
        bellman_loss: TD 誤差に基づく Bellman 損失。
            観測遷移に対して Q 推定を整合させる基本損失。
        conservative_loss: CQL 風の保守的損失。
            未観測行動の過大評価を抑える正則化項。
    """

    total_loss: Array
    bellman_loss: Array
    conservative_loss: Array


@dataclass(frozen=True)
class TrainingTrace:
    """可視化向けに学習履歴をまとめて保持するクラス。

    Attributes:
        metrics: 各 training step の損失履歴。
        positions: 1次元状態の可視化用グリッド。形状は ``[grid_size]``。
        policy_actions: 各 step の greedy 行動マップ。形状は ``[steps, grid_size]``。
            値は行動ID（0=左, 1=右）で、ステップごとの方策変化を追跡できる。
    """

    metrics: list[TrainMetrics]
    positions: Array
    policy_actions: Array


Params = list[dict[str, Array]]


def _init_mlp_params(
    key: Array,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> Params:
    """2 層 MLP（Q ネットワーク）のパラメータを初期化する。

    Args:
        key: JAX の乱数キー。
        input_dim: 入力次元（状態次元）。
        hidden_dim: 隠れ層ユニット数。
        output_dim: 出力次元（行動数）。

    Returns:
        Params: 2 層線形層の重みとバイアスを持つパラメータ構造。
    """
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (input_dim, hidden_dim)) * jnp.sqrt(2.0 / input_dim)
    b1 = jnp.zeros((hidden_dim,))
    w2 = jax.random.normal(k2, (hidden_dim, output_dim)) * jnp.sqrt(2.0 / hidden_dim)
    b2 = jnp.zeros((output_dim,))
    return [{"w": w1, "b": b1}, {"w": w2, "b": b2}]


def q_values(params: Params, observations: Array) -> Array:
    """状態から全行動の Q 値を一括計算する。

    Args:
        params: Q ネットワークのパラメータ。
        observations: 状態バッチ。形状は ``[batch, obs_dim]``。

    Returns:
        Array: 全行動の Q 値。形状は ``[batch, n_actions]``。
    """
    hidden = jnp.tanh(observations @ params[0]["w"] + params[0]["b"])
    return hidden @ params[1]["w"] + params[1]["b"]


def _target_values(target_params: Params, batch: OfflineBatch, gamma: float) -> Array:
    """TD ターゲット値を計算する。

    計算式は以下。
    ``r + gamma * (1 - done) * max_a' Q_target(s', a')``

    Args:
        target_params: ターゲット Q ネットワークのパラメータ。
        batch: 学習に使う遷移ミニバッチ。
        gamma: 割引率。

    Returns:
        Array: 各サンプルの TD ターゲット。形状は ``[batch]``。
    """
    next_q = q_values(target_params, batch.next_observations)
    next_v = jnp.max(next_q, axis=1)
    return batch.rewards + gamma * (1.0 - batch.dones) * next_v


def _take_action_values(all_q: Array, actions: Array) -> Array:
    """全行動 Q 値から、実行行動に対応する Q 値を抽出する。

    Args:
        all_q: 全行動の Q 値。形状は ``[batch, n_actions]``。
        actions: 実行行動ID。形状は ``[batch]``。

    Returns:
        Array: 実行行動に対応する Q 値。形状は ``[batch]``。
    """
    row_ids = jnp.arange(all_q.shape[0])
    return all_q[row_ids, actions]


def compute_losses(
    params: Params,
    target_params: Params,
    batch: OfflineBatch,
    gamma: float,
    cql_alpha: float,
) -> TrainMetrics:
    """Bellman 損失と CQL 風保守的損失を計算する。

    保守的項は次式で計算する。
    ``logsumexp_a Q(s, a) - Q(s, a_dataset)``

    この項により、データで観測されにくい行動の Q 値が過大になりにくくなる。

    Args:
        params: 学習対象（オンライン）Q ネットワークのパラメータ。
        target_params: ターゲット Q ネットワークのパラメータ。
        batch: 学習に使う遷移ミニバッチ。
        gamma: 割引率。
        cql_alpha: 保守的項の重み係数。

    Returns:
        TrainMetrics: 総損失、Bellman 損失、保守的損失。
    """
    all_q = q_values(params, batch.observations)
    q_taken = _take_action_values(all_q, batch.actions)
    td_target = _target_values(target_params, batch, gamma)

    bellman_loss = jnp.mean((q_taken - jax.lax.stop_gradient(td_target)) ** 2)
    conservative_loss = jnp.mean(jax.nn.logsumexp(all_q, axis=1) - q_taken)
    total_loss = bellman_loss + cql_alpha * conservative_loss

    return TrainMetrics(
        total_loss=total_loss,
        bellman_loss=bellman_loss,
        conservative_loss=conservative_loss,
    )


def _soft_update(target_params: Params, online_params: Params, tau: float) -> Params:
    """Polyak 平均でターゲットパラメータを更新する。

    Args:
        target_params: 更新前のターゲットネットワークパラメータ。
        online_params: 最新のオンラインネットワークパラメータ。
        tau: 補間係数。0 に近いほど更新はゆっくりになる。

    Returns:
        Params: 更新後のターゲットネットワークパラメータ。
    """
    return jax.tree_util.tree_map(
        lambda t, o: (1.0 - tau) * t + tau * o, target_params, online_params
    )


def _train_step(
    params: Params,
    target_params: Params,
    opt_state: optax.OptState,
    batch: OfflineBatch,
    gamma: float,
    cql_alpha: float,
    tau: float,
    optimizer: optax.GradientTransformation,
) -> tuple[Params, Params, optax.OptState, TrainMetrics]:
    """オフラインデータのミニバッチで 1 回分の学習更新を行う。

    Args:
        params: 現在のオンライン Q ネットワークパラメータ。
        target_params: 現在のターゲット Q ネットワークパラメータ。
        opt_state: Optax のオプティマイザ状態。
        batch: 固定データセットから抽出したミニバッチ。
        gamma: 割引率。
        cql_alpha: 保守的項の重み係数。
        tau: ターゲット更新係数。
        optimizer: Optax オプティマイザ。

    Returns:
        tuple[Params, Params, optax.OptState, TrainMetrics]:
            更新後のオンラインパラメータ、更新後ターゲットパラメータ、
            更新後オプティマイザ状態、学習メトリクス。
    """

    def loss_fn(current_params: Params) -> Array:
        return compute_losses(
            current_params, target_params, batch, gamma=gamma, cql_alpha=cql_alpha
        ).total_loss

    grads = jax.grad(loss_fn)(params)
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    next_params = optax.apply_updates(params, updates)
    next_target_params = _soft_update(target_params, next_params, tau=tau)
    metrics = compute_losses(
        next_params, next_target_params, batch, gamma=gamma, cql_alpha=cql_alpha
    )
    return next_params, next_target_params, next_opt_state, metrics


def sample_batch(dataset: OfflineBatch, key: Array, batch_size: int) -> OfflineBatch:
    """固定データセットからランダムミニバッチを作成する。

    Args:
        dataset: オフライン遷移データ全体。
        key: サンプリングに使う乱数キー。
        batch_size: 抽出サンプル数。

    Returns:
        OfflineBatch: 指定サイズで抽出した遷移バッチ。
    """
    n = dataset.observations.shape[0]
    indices = jax.random.randint(key, (batch_size,), minval=0, maxval=n)
    return OfflineBatch(
        observations=dataset.observations[indices],
        actions=dataset.actions[indices],
        rewards=dataset.rewards[indices],
        next_observations=dataset.next_observations[indices],
        dones=dataset.dones[indices],
    )


def generate_toy_offline_dataset(
    *,
    seed: int = 0,
    n_episodes: int = 100,
    horizon: int = 25,
) -> OfflineBatch:
    """1 次元移動タスクのオフライン遷移データを生成する。

    環境仕様:
    - 状態: 位置（範囲は ``[-1, 1]``）
    - 行動: 0=左移動, 1=右移動
    - 報酬: ゴール ``+0.8`` への距離の負値

    ふるまい方策は「通常はゴール方向へ進むが、一定確率でランダム行動する」
    ため、品質が混ざったオフラインログが得られる。

    Args:
        seed: データ生成の乱数シード。
        n_episodes: 生成するエピソード数。
        horizon: 1 エピソードの最大ステップ数。

    Returns:
        OfflineBatch: 学習に利用可能な固定遷移データセット。
    """
    rng = np.random.default_rng(seed)
    goal = 0.8

    obs_list: list[float] = []
    act_list: list[int] = []
    rew_list: list[float] = []
    next_obs_list: list[float] = []
    done_list: list[float] = []

    for _ in range(n_episodes):
        state = float(rng.uniform(-1.0, 1.0))
        for _ in range(horizon):
            # ふるまい方策: 基本はゴール方向だが、30%の確率でランダム行動。
            greedy_action = 1 if state < goal else 0
            if rng.random() < 0.3:
                action = int(rng.integers(0, 2))
            else:
                action = greedy_action

            direction = -1.0 if action == 0 else 1.0
            noise = float(rng.normal(0.0, 0.02))
            next_state = float(np.clip(state + 0.12 * direction + noise, -1.0, 1.0))

            reward = -abs(goal - next_state)
            done = float(abs(goal - next_state) < 0.05)

            obs_list.append(state)
            act_list.append(action)
            rew_list.append(reward)
            next_obs_list.append(next_state)
            done_list.append(done)

            state = next_state
            if done > 0.0:
                break

    observations = jnp.asarray(np.asarray(obs_list, dtype=np.float32)[:, None])
    actions = jnp.asarray(np.asarray(act_list, dtype=np.int32))
    rewards = jnp.asarray(np.asarray(rew_list, dtype=np.float32))
    next_observations = jnp.asarray(np.asarray(next_obs_list, dtype=np.float32)[:, None])
    dones = jnp.asarray(np.asarray(done_list, dtype=np.float32))

    return OfflineBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones,
    )


def train_offline_q_learning(
    dataset: OfflineBatch,
    *,
    seed: int = 0,
    config: TrainConfig | None = None,
) -> tuple[Params, list[TrainMetrics]]:
    """固定データのみを使って保守的オフラインQ学習を実行する。

    Args:
        dataset: 事前収集済みのオフライン遷移データ。
        seed: 学習時の乱数シード（ミニバッチ抽出などに使用）。
        config: 学習設定。``None`` の場合は既定値 ``TrainConfig()`` を使う。

    Returns:
        tuple[Params, list[TrainMetrics]]:
            学習済みパラメータと、各ステップのメトリクス履歴。
    """
    cfg = config or TrainConfig()
    key = jax.random.PRNGKey(seed)

    params = _init_mlp_params(
        key, input_dim=cfg.obs_dim, hidden_dim=cfg.hidden_dim, output_dim=cfg.n_actions
    )
    target_params = _init_mlp_params(
        key, input_dim=cfg.obs_dim, hidden_dim=cfg.hidden_dim, output_dim=cfg.n_actions
    )
    optimizer = optax.adam(cfg.learning_rate)
    opt_state = optimizer.init(params)

    history: list[TrainMetrics] = []

    for _ in range(cfg.train_steps):
        key, sample_key = jax.random.split(key)
        batch = sample_batch(dataset, sample_key, cfg.batch_size)
        params, target_params, opt_state, metrics = _train_step(
            params,
            target_params,
            opt_state,
            batch,
            gamma=cfg.gamma,
            cql_alpha=cfg.cql_alpha,
            tau=cfg.tau,
            optimizer=optimizer,
        )
        history.append(metrics)

    return params, history


def greedy_action(params: Params, observation: Array) -> int:
    """単一状態での greedy 行動（argmax Q）を返す。

    Args:
        params: 学習済み Q ネットワークパラメータ。
        observation: 単一状態ベクトル。形状は ``[obs_dim]``。

    Returns:
        int: 選択された行動ID。
    """
    q = q_values(params, observation[None, :])
    return int(jnp.argmax(q[0]))


def greedy_actions_on_1d_grid(
    params: Params,
    *,
    x_min: float = -1.0,
    x_max: float = 1.0,
    grid_size: int = 121,
) -> tuple[Array, Array]:
    """1次元グリッド上の各状態で greedy 行動を評価する。

    Args:
        params: 学習済み Q ネットワークパラメータ。
        x_min: 可視化グリッドの最小位置。
        x_max: 可視化グリッドの最大位置。
        grid_size: グリッド点数。

    Returns:
        tuple[Array, Array]:
            ``(positions, actions)`` を返す。
            - positions: 位置グリッド。形状は ``[grid_size]``。
            - actions: 各位置での greedy 行動。形状は ``[grid_size]``。
    """
    positions = jnp.linspace(x_min, x_max, grid_size, dtype=jnp.float32)
    observations = positions[:, None]
    q = q_values(params, observations)
    actions = jnp.argmax(q, axis=1)
    return positions, actions


def run_demo(seed: int = 0) -> dict[str, float]:
    """学習の流れを手早く確認するためのデモを実行する。

    Args:
        seed: データ生成と学習で共有する乱数シード。

    Returns:
        dict[str, float]:
            初期損失・最終損失・代表状態での行動を含む辞書。
            学習が進むと ``last_loss`` が ``first_loss`` より小さくなることを
            期待できる。
    """
    dataset = generate_toy_offline_dataset(seed=seed)
    config = TrainConfig(train_steps=300)
    params, history = train_offline_q_learning(dataset, seed=seed, config=config)

    left_state = jnp.asarray([-0.7], dtype=jnp.float32)
    right_state = jnp.asarray([0.95], dtype=jnp.float32)

    return {
        "first_loss": float(history[0].total_loss),
        "last_loss": float(history[-1].total_loss),
        "action_left_state": float(greedy_action(params, left_state)),
        "action_right_state": float(greedy_action(params, right_state)),
    }


def train_offline_q_learning_with_trace(
    dataset: OfflineBatch,
    *,
    seed: int = 0,
    config: TrainConfig | None = None,
    grid_size: int = 121,
) -> tuple[Params, TrainingTrace]:
    """学習しながら方策の1次元推移を記録する。

    Args:
        dataset: 事前収集済みのオフライン遷移データ。
        seed: 学習時の乱数シード。
        config: 学習設定。``None`` の場合は ``TrainConfig()`` を使う。
        grid_size: 1次元方策可視化で使う状態グリッド点数。

    Returns:
        tuple[Params, TrainingTrace]:
            学習済みパラメータと、可視化用トレース情報。
    """
    cfg = config or TrainConfig()
    key = jax.random.PRNGKey(seed)

    params = _init_mlp_params(
        key, input_dim=cfg.obs_dim, hidden_dim=cfg.hidden_dim, output_dim=cfg.n_actions
    )
    target_params = _init_mlp_params(
        key, input_dim=cfg.obs_dim, hidden_dim=cfg.hidden_dim, output_dim=cfg.n_actions
    )
    optimizer = optax.adam(cfg.learning_rate)
    opt_state = optimizer.init(params)

    history: list[TrainMetrics] = []
    positions = jnp.linspace(-1.0, 1.0, grid_size, dtype=jnp.float32)
    action_snapshots: list[Array] = []

    for _ in range(cfg.train_steps):
        key, sample_key = jax.random.split(key)
        batch = sample_batch(dataset, sample_key, cfg.batch_size)
        params, target_params, opt_state, metrics = _train_step(
            params,
            target_params,
            opt_state,
            batch,
            gamma=cfg.gamma,
            cql_alpha=cfg.cql_alpha,
            tau=cfg.tau,
            optimizer=optimizer,
        )
        history.append(metrics)

        # 各 step で 1次元状態全体の greedy 行動を保存し、方策変化を追跡する。
        q = q_values(params, positions[:, None])
        action_snapshots.append(jnp.argmax(q, axis=1))

    trace = TrainingTrace(
        metrics=history,
        positions=positions,
        policy_actions=jnp.stack(action_snapshots, axis=0),
    )
    return params, trace


def _require_matplotlib() -> Any:
    """matplotlib を遅延 import し、未導入時は案内付きで例外を送出する。"""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - 実環境依存分岐
        raise ModuleNotFoundError(
            "matplotlib が見つかりません。`uv add matplotlib` で導入してください。"
        ) from exc
    return plt


def plot_learning_curve(
    metrics: list[TrainMetrics],
    *,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """学習曲線（損失推移）を matplotlib で描画する。

    Args:
        metrics: 各 step の学習メトリクス。
        save_path: 画像保存先。``None`` の場合は保存しない。
        show: ``True`` の場合に画面表示する。
    """
    if not metrics:
        raise ValueError("metrics が空です。可視化する学習履歴がありません。")

    plt = _require_matplotlib()
    steps = np.arange(len(metrics))
    total = np.asarray([float(m.total_loss) for m in metrics], dtype=np.float32)
    bellman = np.asarray([float(m.bellman_loss) for m in metrics], dtype=np.float32)
    conservative = np.asarray([float(m.conservative_loss) for m in metrics], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, total, label="total_loss")
    ax.plot(steps, bellman, label="bellman_loss")
    ax.plot(steps, conservative, label="conservative_loss")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title("Offline RL Learning Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)

    if show:
        plt.show()
    plt.close(fig)


def plot_1d_policy_dynamics(
    trace: TrainingTrace,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """training step ごとの 1次元方策変化をヒートマップで可視化する。

    Args:
        trace: ``train_offline_q_learning_with_trace`` が返す学習トレース。
        save_path: 画像保存先。``None`` の場合は保存しない。
        show: ``True`` の場合に画面表示する。
    """
    plt = _require_matplotlib()
    policy = np.asarray(trace.policy_actions, dtype=np.int32)
    x = np.asarray(trace.positions, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        policy,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        interpolation="nearest",
        extent=[float(x[0]), float(x[-1]), 0, policy.shape[0]],
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("position (1D state)")
    ax.set_ylabel("training step")
    ax.set_title("1D Greedy Policy Dynamics (0=left, 1=right)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("greedy action")
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)

    if show:
        plt.show()
    plt.close(fig)


def run_visualization_demo(
    *,
    seed: int = 0,
    train_steps: int = 300,
    output_dir: str = "artifacts",
) -> dict[str, str]:
    """学習曲線と 1次元方策推移をまとめて出力するデモ関数。

    Args:
        seed: 乱数シード。
        train_steps: 学習ステップ数。
        output_dir: 可視化画像を保存するディレクトリ。

    Returns:
        dict[str, str]: 生成した画像パスを含む辞書。
    """
    dataset = generate_toy_offline_dataset(seed=seed)
    config = TrainConfig(train_steps=train_steps)
    _, trace = train_offline_q_learning_with_trace(dataset, seed=seed, config=config)

    output = Path(output_dir)
    learning_curve_path = output / "learning_curve.png"
    policy_dynamics_path = output / "policy_dynamics.png"

    plot_learning_curve(trace.metrics, save_path=str(learning_curve_path), show=False)
    plot_1d_policy_dynamics(trace, save_path=str(policy_dynamics_path), show=False)

    return {
        "learning_curve_path": str(learning_curve_path),
        "policy_dynamics_path": str(policy_dynamics_path),
    }


if __name__ == "__main__":
    demo_result = run_demo(seed=0)
    print("offline_rl demo result:")
    for name, value in demo_result.items():
        print(f"{name}: {value}")
    try:
        paths = run_visualization_demo(seed=0, train_steps=200, output_dir="artifacts")
        print("saved figures:")
        for name, path in paths.items():
            print(f"{name}: {path}")
    except ModuleNotFoundError as exc:
        print(exc)
