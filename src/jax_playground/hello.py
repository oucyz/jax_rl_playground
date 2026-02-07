import jax
import jax.numpy as jnp
import optax
from flax import nnx


def log2(x):
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.0)
    return ln_x / ln_2


class LinearModel(nnx.Module):
    def __init__(self, n_hidden, rngs: nnx.Rngs):
        super().__init__()
        self.linear = nnx.Linear(4, n_hidden, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


def main():
    print(jax.make_jaxpr(log2)(2.0))
    model = LinearModel(n_hidden=8, rngs=nnx.Rngs(0))
    output = model(jnp.ones((1, 4)))
    print(output)

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=0.01, weight_decay=0.001))
    print(optimizer.target)


if __name__ == "__main__":
    main()
