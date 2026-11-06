import numpy as np 
import jax.numpy as jnp
import jax 
import json
import time
from scipy.optimize import minimize
from src.solvers.impes_jax import calc_prod_jax
import matplotlib.pyplot as plt
from collections import namedtuple
import os
import logging

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

path = 'inputs/impes_input.json'
with open(path, 'r') as p:
    parameters_dict = json.load(p)

params_immutable = namedtuple('params_immutable', parameters_dict.keys())
parameters = params_immutable(**parameters_dict)

def profit(t_inj):
    
    # t_inj = t_inj_arr[0] 
    t_inj = jnp.squeeze(t_inj)

    prod = calc_prod_jax(t_inj,parameters)

    # custo goma xantana 48,90R$/kg , considerando 500ppm de polimero (0.0005kg), 
    # com uma vazao de entrada de 685ft³/dia, 685ft³ é aproximadamente 19000kg, então 
    # usamos 9,5kg de polimero por dia, totalizando 464,55R$/dia 
    # barril petroleo 5,6146ft³, valendo aproximadamente 388,29 R$

    # f = (prod/volume)*price - (lmbd*t_inj)
    # f  = prod - 2*t_inj

    price, volume, lmbd = 388.29, 5.6146, 464.55
    f = (prod / volume) * price - lmbd * t_inj     # R$
    return -f / 1e6                                 # milhões de R$
    
    # return -f

jacobian_rev = jax.jit(jax.jacfwd(profit))
profit_jit = jax.jit(profit)

# @jax.jit
# def fwd(t_inj):
#     value, grad = jax.jvp(profit,(t_inj,),(1.0,))
#     return value, grad

# f_jax_jit = jax.jit(jax.value_and_grad(profit,argnums=0))
# fun = jax.value_and_grad(profit,argnums=0)

def F(t_inj_arr):
    t_inj = t_inj_arr[0]
    # value, grad = f_jax_jit(t_inj)
    # value, grad = fwd(t_inj)
    value = profit_jit(t_inj)
    grad = jacobian_rev(t_inj)
    return float(value), np.array([float (grad)])

cont = 0 

def optmz_jax(): 

    def callback(xk):
        global cont
        val , grad= F(xk)
        print(" It : {} , t_inj = {} , F = {} , G = {} ".format(cont,xk[0],-val,grad[0]))
        cont+=1

    sol = minimize(F,x0 = 10.00,  
                   method='L-BFGS-B', 
                   bounds=[(0.0,parameters.tf)], 
                   jac=True,
                   options={'ftol':1e-9,'gtol':1e-9}, 
                   callback = callback)
    print(sol)
    return sol

# run optimization
start = time.perf_counter()
sol = optmz_jax()
end = time.perf_counter()
print(" elapsed time:", end - start)
print(jax.devices())

#
# brute-force landscape x optimized solution
#
t_inj = jnp.linspace(0.0, parameters.tf,100)
varredura = jnp.zeros_like(t_inj)
for i in range(len(t_inj)):
    varredura = varredura.at[i].set(profit(t_inj[i]))

plt.figure(figsize=(12,8))
plt.scatter(sol.x[0],-sol.fun,label ='Minimize',marker = '^', color= 'red')
plt.plot(t_inj,-varredura,'--o', label = 'Varredura')
plt.xlabel("Tempo de injeção")
plt.ylabel("Lucro")
plt.grid(True)
plt.legend()
plt.show()