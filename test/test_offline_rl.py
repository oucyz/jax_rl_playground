import jax.numpy as jnp

from jax_playground.offline_rl import (
    TrainConfig,
    generate_toy_offline_dataset,
    greedy_action,
    train_offline_q_learning,
)


def test_generate_toy_offline_dataset_shapes() -> None:
    dataset = generate_toy_offline_dataset(seed=7, n_episodes=10, horizon=8)

    n = dataset.observations.shape[0]
    assert n > 0
    assert dataset.observations.shape[1] == 1
    assert dataset.next_observations.shape == dataset.observations.shape
    assert dataset.actions.shape == (n,)
    assert dataset.rewards.shape == (n,)
    assert dataset.dones.shape == (n,)


def test_offline_training_reduces_total_loss() -> None:
    dataset = generate_toy_offline_dataset(seed=11, n_episodes=120, horizon=20)
    config = TrainConfig(train_steps=200, batch_size=128)

    _, history = train_offline_q_learning(dataset, seed=11, config=config)

    first_loss = float(history[0].total_loss)
    last_loss = float(history[-1].total_loss)

    assert last_loss < first_loss


def test_greedy_policy_moves_toward_goal() -> None:
    dataset = generate_toy_offline_dataset(seed=23, n_episodes=150, horizon=20)
    config = TrainConfig(train_steps=250, batch_size=128)

    params, _ = train_offline_q_learning(dataset, seed=23, config=config)

    # 左側では右移動(action=1)が望ましい
    assert greedy_action(params, jnp.asarray([-0.7], dtype=jnp.float32)) == 1
    # 中央付近でもゴール(+0.8)方向である右移動(action=1)を選べること
    assert greedy_action(params, jnp.asarray([0.0], dtype=jnp.float32)) == 1
