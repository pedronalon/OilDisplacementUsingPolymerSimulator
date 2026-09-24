import json
import time

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu

from src.solvers.impes_jax import calc_prod_jax
from collections import namedtuple 

# jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)




path = '/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/inputs/impes_input.json'
with open(path, 'r') as p:
    parameters_dict = json.load(p)

params_immutable = namedtuple('params_immutable', parameters_dict.keys())
parameters = params_immutable(**parameters_dict)

TF = float(parameters.tf) 

def profit(t_inj):
    prod = calc_prod_jax(t_inj, parameters)
    price = 388.29
    volume = 5.6156
    lmbd = 464.55
    f = (prod / volume) * price - (lmbd * t_inj)
    return -f / 1e7


# ---- Bound handling: t_inj = TF * sigmoid(z) keeps t_inj in (0, TF) ----
def z_to_t(z):
    return TF * jax.nn.sigmoid(z)

def t_to_z(t):
    p = t / TF
    return jnp.log(p) - jnp.log1p(-p)   # logit

def objective(z):
    # z has shape (1,), profit expects a scalar
    return profit(z_to_t(z)[0])


def run_opt(z0, fun, opt, max_iter, gtol, ftol):
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        z, state, _ = carry
        value, grad = value_and_grad_fun(z, state=state)   # value/grad at current z
        updates, state = opt.update(
            grad, state, z, value=value, grad=grad, value_fn=fun
        )
        z = optax.apply_updates(z, updates)
        jax.debug.print(
            " It : {it} , t_inj = {t} , F = {f} , |G| = {g}",
            it=otu.tree_get(state, 'count'),
            t=z_to_t(z)[0],
            f=-otu.tree_get(state, 'value'),
            g=otu.tree_l2_norm(otu.tree_get(state, 'grad')),
        )
        return z, state, value   # carry the previous value for the ftol check

    def keep_going(carry):
        _, state, f_prev = carry
        it = otu.tree_get(state, 'count')
        f_new = otu.tree_get(state, 'value')      # value at the new iterate
        g_new = otu.tree_get(state, 'grad')

        # scipy-style tolerances
        gnorm = jnp.max(jnp.abs(g_new))                                   # gtol
        rel_df = jnp.abs(f_prev - f_new) / jnp.maximum(
            jnp.maximum(jnp.abs(f_prev), jnp.abs(f_new)), 1.0)            # ftol

        converged = (gnorm <= gtol) | (rel_df <= ftol)
        return (it == 0) | ((it < max_iter) & ~converged)

    init_carry = (z0, opt.init(z0), jnp.array(jnp.inf, dtype=z0.dtype))
    z_final, state_final, _ = jax.lax.while_loop(keep_going, step, init_carry)
    return z_final, state_final


solver = optax.lbfgs()   # default: zoom linesearch satisfying strong Wolfe

run_opt_jit = jax.jit(
    lambda z0: run_opt(z0, objective, solver, max_iter=100, gtol=1e-6, ftol=1e-16)
)

t0 = 10.0
z0 = jnp.array([t_to_z(t0)])

start = time.perf_counter()
z_opt, state = run_opt_jit(z0)
z_opt.block_until_ready()          # JAX is async; wait before stopping the timer
end = time.perf_counter()

t_opt = z_to_t(z_opt)[0]
print(f"t_inj* = {float(t_opt):.6f}")
print(f"Profit = {float(-objective(z_opt)) * 1e7:.4f}")
print(f"Iterations = {int(otu.tree_get(state, 'count'))}")
print("time (includes compilation):", end - start)
print(jax.devices())