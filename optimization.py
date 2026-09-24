import numpy as np 
import jax.numpy as jnp
import jax 
import json
import time
from scipy.optimize import minimize
from src.solvers.impes_numpy import calc_prod 
from src.solvers.impes_jax import calc_prod_jax
import matplotlib.pyplot as plt

# jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

cont = 0
path = '/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/inputs/impes_input.json'

with open(path,'r') as p:
    parameters = json.load(p)

def profit(t_inj):
    
    # t_inj = t_inj_arr[0] 
    

    prod = calc_prod_jax(t_inj,parameters)
   

    # custo goma xantana 48,90R$/kg , considerando 500ppm de polimero (0.0005kg), 
    # com uma vazao de entrada de 685ft³/dia, 685ft³ é aproximadamente 19000kg, então 
    # usamos 9,5kg de polimero por dia, totalizando 464,55R$/dia 
    # barril petroleo 5,6146ft³, valendo aproximadamente 388,29 R$
   
    price = 388.29
    volume = 5.6156
    lmbd = 464.55


    f = (prod/volume)*price - (lmbd*t_inj)

    return (-f/1e7)

# # jacobian_rev = jax.jit(jax.jacfwd(profit))
# # profit_jit = jax.jit(profit)
# @jax.jit
# def fwd(t_inj):
#     value, grad = jax.jvp(profit,(t_inj,),(1.0,))
#     return value, grad


f_jax_jit = jax.jit(jax.value_and_grad(profit,argnums=0))
fun = jax.value_and_grad(profit,argnums=0)


def F(t_inj_arr):
    t_inj = t_inj_arr[0]
    
    value, grad = fun(t_inj)

    # value, grad = fwd(t_inj)
    # value = profit(t_inj)
    # grad = jacobian_rev(t_inj)


    return float(value), np.array([float (grad)])


def optmz_jax(): 

    
    def callback(xk):
        global cont

        val , grad= F(xk)
        print(" It : {} , t_inj = {} , F = {} , G = {} ".format(cont,xk[0],-val,grad[0]))
        cont+=1

    sol = minimize(F,x0 =10.00,  
                   method='L-BFGS-B', 
                   bounds=[(0.0,parameters.get('tf'))], 
                   jac=True,
                   options={'ftol':1e-16,'gtol':1e-6, 'disp' : True}, 
                   callback = callback)

    print(sol)
    return sol


start = time.perf_counter()

sol = optmz_jax()

end = time.perf_counter()
print("time:", end - start)
print(jax.devices())

# t_inj = jnp.arange(0,1400,25)
# varredura = jnp.zeros_like(t_inj)

# for i in range(len(t_inj)):
    
#     varredura = varredura.at[i].set(profit(t_inj[i]))

# plt.figure(figsize=(12,8))
# plt.scatter(sol.x[0],-sol.fun,label ='Minimize', color= 'red')
# plt.plot(t_inj,-varredura,'--o', label = 'Brute Force')
# plt.xlabel("Injection Time")
# plt.ylabel("Profit")
# plt.grid(True)
# plt.legend()
# plt.show()

