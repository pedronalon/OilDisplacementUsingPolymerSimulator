import jax.numpy as jnp
import jax
import pandas as pd

def jax_simulator(t_inj, parameters, is_optimizing=False):

    Swc = parameters.get("Swc") 
    Sor = parameters.get("Sor") 

    krw0 = parameters.get("krw0") 
    kro0 = parameters.get("kro0") 
    k = parameters.get("k")

    nw = parameters.get("nw")  
    no = parameters.get("no") 

    mu_w = parameters.get("mu_w") 
    mu_o = parameters.get("mu_o") 

    phi = parameters.get("phi") 
    a = parameters.get("a") 
    q = parameters.get("q") 

    ti = parameters.get("ti") 
    tf = parameters.get("tf") 

    Li = parameters.get("Li")
    Lf = parameters.get("Lf") 
    M = parameters.get("M")
    M_0 = M[0] 

    def Krw(Sw):
        sn = jnp.clip((Sw - Swc) / (1 - Swc - Sor), 0.0, 1.0)
        kr = krw0 * (sn)**nw
        return kr

    def Kro(Sw):
        jnp.clip(Sw, Sor, 1 - Swc)
        sn = jnp.clip((Sw - Swc) / (1 - Swc - Sor), 0.0, 1.0)
        kr = kro0 * (1 - sn)**no 
        return kr

    def linear_interpolation(c, v, x):
        return jnp.interp(x, c, v)

    dx = (Lf-Li)/(M_0-1)
    v = q/a 
    vp = phi*a*dx
    
    dt = 0.0800827936386817
    N = int(tf/dt)

    t = jnp.linspace(ti, tf, N-1)
    x = jnp.linspace(Li, Lf, M_0)

    Sw = jnp.zeros(M_0).at[:].set(Swc+0.001)
    C = jnp.zeros(M_0)
    p = jnp.zeros(M_0)
    T = jnp.zeros(M_0)
    
    qw = jnp.zeros(M_0-1)
    T_inter = jnp.zeros(M_0 - 1)
    T_matrix = jnp.zeros((M_0 - 1, M_0 - 1))
    
    p_right = 0.0

    def solve_pressure(pressure, Sw_current, C_current): 
        
        qt = jnp.zeros(M_0-1)
        lmbd = jnp.zeros(M_0)
        concentration = jnp.array([0.0, 100.0, 300.0, 500.0])
        viscosity = jnp.array([0.5, 1.5, 4.0, 7.8])

        lmbd_w = (k * Krw(Sw_current)) / (linear_interpolation(concentration, viscosity, C_current))
        lmbd_o = (k * Kro(Sw_current)) / mu_o 
        lmbd = lmbd_o + lmbd_w

        fw_up = lmbd_w / (lmbd_o + lmbd_w)

        T = (6.3283e-3 * lmbd[:] * a) / dx
                
        main_diag = jnp.zeros(M_0-1)
        lower_diag = jnp.zeros(M_0-1)
        upper_diag = jnp.zeros(M_0-1)

        T_right = jnp.zeros(M_0-1)
        T_left = jnp.zeros(M_0-1)
                
        T_right = (2 * T[:-1] * T[1:]) / (T[:-1] + T[1:])
        T_left_temp = (2 * T[:-2] * T[1:-1]) / (T[:-2] + T[1:-1])
        T_left = T_left.at[1:].set(T_left_temp)
            
        T_inter = T_right

        main_diag = main_diag.at[0].set(T_right[0])
        upper_diag = upper_diag.at[0].set(-T_right[0])
        lower_diag = lower_diag.at[0].set(0.0)
        qt = qt.at[0].set(q)
        
        main_diag = main_diag.at[-1].set(T_left[-1] + T_right[-1])
        lower_diag = lower_diag.at[-1].set(-T_left[-1])
        upper_diag = upper_diag.at[-1].set(0.0)
        qt = qt.at[-1].set(T_right[-1] * p_right) 

        main_diag = main_diag.at[1:-1].set(T_left[1:-1] + T_right[1:-1])
        lower_diag = lower_diag.at[1:-1].set(-T_left[1:-1])
        upper_diag = upper_diag.at[1:-1].set(-T_right[1:-1])
        qt = qt.at[1:-1].set(0.0)

        qt_matrix = qt[:, None]
        
        solver = jax.lax.linalg.tridiagonal_solve(lower_diag, main_diag, upper_diag, qt_matrix)
        solver = solver.squeeze()

        pressure = pressure.at[:-1].set(solver)
        pressure = pressure.at[-1].set(p_right)

        Qt = -T_inter[:] * (pressure[1:] - pressure[:-1])
        Qw = fw_up[:-1] * Qt

        return pressure, Qw, Qt

    def impes_step(carry, t_step):
        
        Sw, C, p = carry
        
        p, qw, Qt = solve_pressure(p, Sw, C)

        Sw_old = Sw.copy()
        
        Sw_0 = Sw[0] + (dt / vp) * (q - qw[0])
        Sw = Sw.at[0].set(Sw_0)

        Sw_internal = Sw[1:-1] + (dt / vp) * (qw[:-1] - qw[1:])
        Sw = Sw.at[1:-1].set(Sw_internal)

        Sw = Sw.at[-1].set(Sw_old[-1])

        alpha = (qw*dt)/(vp)
        alpha_inj = (q*dt)/(vp)

        b = 1.0
        poly_inj = jax.nn.sigmoid(b*(t_inj-t_step)) * 500.00
        
        C_f = ((Sw_old[-1] - alpha[-1]) * C[-1] + alpha[-1] * C[-2]) / Sw[-1]
        C = C.at[-1].set(C_f)
        
        C_inernal = ((Sw_old[1:-1] - alpha[1:]) * C[1:-1] + alpha[:-1] * C[:-2]) / Sw[1:-1]
        C = C.at[1:-1].set(C_inernal)

        C_0 = ((Sw_old[0] - alpha[0]) * C[0] + alpha_inj * poly_inj) / Sw[0]
        C = C.at[0].set(C_0)

        Sw = jnp.clip(Sw, Swc, 1.0 - Sor)

        prod_o = Qt[-1] - qw[-1]

        if is_optimizing:
            return (Sw, C, p), prod_o
        else:
            return (Sw, C, p), (Sw, p, C, prod_o)

    if is_optimizing:
        impes_sol, prod_o_hist = jax.lax.scan(impes_step, (Sw, C, p), t)
        return prod_o_hist, dt
    else:
        impes_sol, (Sw_hist, p_hist, C_hist, prod_o_hist) = jax.lax.scan(impes_step, (Sw, C, p), t)
        return (Sw_hist, p_hist, C_hist, prod_o_hist), dt, t, x, N


def jax_solver(parameters):
    t_inj = parameters.get("t_inj")

    (Sw_hist, p_hist, C_hist, prod_o_hist), dt, t, x, N = jax_simulator(t_inj, parameters, is_optimizing=False)
    prod = jnp.cumsum(prod_o_hist)*dt
    plot_values = [int(N/4)-1, int(N/2)-1, int(3*N/4)-1]

    data = {
            'x' : x ,

            'Sw_1' : Sw_hist[plot_values[0],:],
            'p_1' : p_hist[plot_values[0],:],

            'Sw_2' : Sw_hist[plot_values[1],:],
            'p_2' : p_hist[plot_values[1],:],

            'Sw_3' : Sw_hist[plot_values[2],:], 
            'p_3' : p_hist[plot_values[2],:], 

            'Sw_4' : Sw_hist[-1,:], 
            'p_4' : p_hist[-1,:] ,
            
            'C_1' : C_hist[plot_values[0],:],
            'C_2' : C_hist[plot_values[1],:],
            'C_3' : C_hist[plot_values[2],:], 
            'C_4' : C_hist[-1,:] , 
        }

    prod_data =  {
            't' : t,
            'prod_1' : prod,
        }

    return pd.DataFrame(data), pd.DataFrame(prod_data)
    

def calc_prod_jax(t_inj, parameters):
    
    prod_o_hist, dt = jax_simulator(t_inj, parameters, is_optimizing=True)
    
    prod_total = jnp.sum(prod_o_hist) * dt
    
    return prod_total