# jax_rl_playground

JAX を使って強化学習を学ぶためのサンプル集です。  
現在は **オフラインRL（Conservative Q-Learning 風）** の最小実装を収録しています。

## このサンプルで学べること

- オフラインRLの基本フロー
- 固定データセットのみで Q 関数を学習する手順
- Bellman 損失に保守的正則化（CQL風）を足す考え方
- JAX + Optax での小規模学習ループ実装

## 実装ファイル

- `/Users/oucyz/repos/jax_rl_playground/src/jax_playground/offline_rl.py`
- `/Users/oucyz/repos/jax_rl_playground/test/test_offline_rl.py`

## クイックスタート

```zsh
# 依存関係を同期
uv sync

# 仮想環境を有効化（任意）
. .venv/bin/activate
```

## 実行例（Python REPL）

```python
from jax_playground.offline_rl import run_demo

result = run_demo(seed=0)
print(result)
# 例:
# {
#   'first_loss': ...,
#   'last_loss': ...,
#   'action_left_state': 1.0,
#   'action_right_state': 1.0,
# }
```

`first_loss` と `last_loss` を比べることで、学習が進んでいるかを確認できます。

## オフラインRLサンプルの流れ

1. `generate_toy_offline_dataset()` で固定データを作る
2. `train_offline_q_learning()` でデータのみを使って学習する
3. `greedy_action()` で学習後方策の行動を確認する

## 品質チェック

```zsh
# フォーマット
make fmt

# Lint + 型チェック
make lint

# テスト
make test
```

## テスト内容

`test/test_offline_rl.py` では、以下を検証しています。

- データセットの形状が期待どおりであること
- 学習で損失が低下すること
- 学習後方策がゴール方向の行動を選べること
