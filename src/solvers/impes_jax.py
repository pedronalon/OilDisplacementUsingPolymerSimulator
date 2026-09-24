import jax.numpy as jnp
import jax
import pandas as pd
import h5py
import numpy as np
import json




@jax.jit(static_argnames=['parameters', 'is_optimizing'])
def jax_simulator(t_inj, parameters, is_optimizing = False):

    Swc = parameters.Swc 
    Sor = parameters.Sor 

    krw0 = parameters.krw0 
    kro0 = parameters.kro0 
    k = parameters.k

    nw = parameters.nw  
    no = parameters.no 

    mu_w = parameters.mu_w 
    mu_o = parameters.mu_o 

    phi = parameters.phi 
    a = parameters.a 
    q = parameters.q 

    ti = parameters.ti 
    tf = parameters.tf 

    Li = parameters.Li
    Lf = parameters.Lf 
    M = parameters.M
    M_0 = parameters.M 
    max_rrf = parameters.max_rrf
    Csf = parameters.Csf

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

    def langmuir(C):
        C_norm = C/500.00
        A = 0.0
        B = 0.0

        d_dc = A/(1 + B*C_norm)**2
        
        return d_dc 

    def impes_step(carry, t_step):
        
        Sw, C, p = carry
        
        p, qw, Qt = solve_pressure(p, Sw, C)

        Sw_old = Sw.copy()
        
        Sw_0 = Sw[0] + (dt / vp) * (q - qw[0])
        Sw = Sw.at[0].set(Sw_0)

        Sw_internal = Sw[1:-1] + (dt / vp) * (qw[:-1] - qw[1:])
        Sw = Sw.at[1:-1].set(Sw_internal)

        Sw = Sw.at[-1].set(Sw_old[-1])

        Sw = jnp.clip(Sw, Swc, 1.0 - Sor)

        # alpha = (qw*dt)/(vp)
        # alpha_inj = (q*dt)/(vp)

        
        b = 1.0
        poly_inj = jax.nn.sigmoid(b*(t_inj-t_step)) * 500.00
        

        C_ads = langmuir(C)
        C_old = C.copy()

        # C_0 = ((Sw_old[0] - alpha[0]) * C[0] + alpha_inj * poly_inj) / Sw[0]

        C_0 = ((C_old[0]*(Sw_old[0] + C_ads[0])) + (dt/vp)*(q * poly_inj - qw[0] * C_old[0]))/(
            Sw[0] + C_ads[0])
        
        C = C.at[0].set(C_0)

        # C_internal = ((Sw_old[1:-1] - alpha[1:]) * C[1:-1] + alpha[:-1] * C[:-2]) / Sw[1:-1]
        C_internal = (C_old[1:-1] * (Sw_old[1:-1] + C_ads[1:-1]) + (dt / vp) * (qw[:-1] * C_old[:-2] - qw[1:] * C_old[1:-1]))/(
            Sw[1:-1] + C_ads[1:-1])
        
        C = C.at[1:-1].set(C_internal)

        # C_f = ((Sw_old[-1] - alpha[-1]) * C[-1] + alpha[-1] * C[-2]) / Sw[-1]
        C_f = (C_old[-1] * (Sw_old[-1] + C_ads[-1]) + (dt / vp) * (qw[-1] * C_old[-2] - qw[-1] * C[-1]))/(
            Sw[-1] + C_ads[-1])

        C = C.at[-1].set(C_f)

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

        V = phi*a*Lf
        pvi = (q*t)/V
        
        
        #massa injetada, corrigir esse print depois 
        # b = 1.0 
        # poly_inj = jax.nn.sigmoid(b * (t_inj - t)) * 500.0
        # mass = jnp.sum(q * poly_inj) * dt

        # print("polymer injected mass: {}".format(mass))
        
        return (Sw_hist, p_hist, C_hist, prod_o_hist), dt, pvi, t, x, N




def generate_xdmf(file_prefix, M_0, t_save, h5_path):
    """Gera o arquivo XDMF já estruturado como um bloco 3D (Hexaedros)"""
    xdmf_content = ['<?xml version="1.0" ?>', '<Xdmf Version="3.0">', '<Domain>', '<Grid GridType="Collection" CollectionType="Temporal">']
    
    for i, t in enumerate(t_save):
        # Dimensions="2 2 M" cria 1 célula no eixo Y e 1 no eixo Z
        grid = f"""
        <Grid Name="Step_{i}" GridType="Uniform">
            <Time Value="{t:.4f}"/>
            <Topology TopologyType="3DRectMesh" Dimensions="2 2 {M_0+1}"/>
            <Geometry GeometryType="VXVYVZ">
                <DataItem Dimensions="2" Format="XML">0.0 1.0</DataItem>
                <DataItem Dimensions="2" Format="XML">0.0 1.0</DataItem>
                <DataItem Dimensions="{M_0+1}" Format="HDF">{h5_path}:/spatial/x_edges</DataItem>
            </Geometry>
            <Attribute Name="Sw" AttributeType="Scalar" Center="Cell">
                <DataItem Dimensions="1 1 {M_0}" Format="HDF">{h5_path}:/spatial/Sw_{i}</DataItem>
            </Attribute>
            <Attribute Name="Pressure" AttributeType="Scalar" Center="Cell">
                <DataItem Dimensions="1 1 {M_0}" Format="HDF">{h5_path}:/spatial/p_{i}</DataItem>
            </Attribute>
            <Attribute Name="Concentration" AttributeType="Scalar" Center="Cell">
                <DataItem Dimensions="1 1 {M_0}" Format="HDF">{h5_path}:/spatial/C_{i}</DataItem>
            </Attribute>
        </Grid>"""
        xdmf_content.append(grid)
        
    xdmf_content.extend(['</Grid>', '</Domain>', '</Xdmf>'])
    
    with open(f"outputs/{file_prefix}.xmf", "w") as f:
        f.write("\n".join(xdmf_content))
    
    with open(f"outputs/{file_prefix}.xmf", "w") as f:
        f.write("\n".join(xdmf_content))

def jax_solver(parameters, dt_save=1.0, file_prefix="sim_output"):
    t_inj = parameters.t_inj

    (Sw_hist, p_hist, C_hist, prod_o_hist), dt, pvi, t, x, N = jax_simulator(t_inj, parameters, is_optimizing = False)
    
    #  dt_save
    save_step = max(1, int(dt_save / dt))
    
    Sw_save = np.array(Sw_hist[::save_step, :])
    p_save = np.array(p_hist[::save_step, :])
    C_save = np.array(C_hist[::save_step, :])
    t_save = np.array(t[::save_step])
    
    
    Li, Lf, M_0 = parameters.Li, parameters.Lf, parameters.M
    dx = (Lf - Li) / (M_0 - 1)
    x_edges = np.linspace(Li - dx/2, Lf + dx/2, M_0 + 1)
    
    #  EXPORTAR HDF5
    h5_path = f"outputs/{file_prefix}.h5"
    
    with h5py.File(h5_path, 'w') as f:
        # Grupo Temporal
        temporal = f.create_group("temporal")
        temporal.create_dataset("t", data=np.array(t))
        temporal.create_dataset("pvi", data=np.array(pvi))
        temporal.create_dataset("prod", data=np.array(jnp.cumsum(prod_o_hist) * dt))
        temporal.create_dataset("Cp", data=np.array(C_hist[:, -1] / 500.0))
        
        # Grupo Espacial
        spatial = f.create_group("spatial")
        spatial.create_dataset("x_edges", data=x_edges)
        spatial.create_dataset("x_centers", data=np.array(x))
        spatial.create_dataset("t_save", data=t_save)
        
        for i in range(len(t_save)):
            spatial.create_dataset(f"Sw_{i}", data=Sw_save[i, :])
            spatial.create_dataset(f"p_{i}", data=p_save[i, :])
            spatial.create_dataset(f"C_{i}", data=C_save[i, :])

    #  EXPORTAR XDMF
    generate_xdmf(file_prefix, M_0, t_save, f"{file_prefix}.h5")
    
    return h5_path


def calc_prod_jax(t_inj, parameters):
    
    prod_o_hist, dt = jax_simulator(t_inj, parameters, is_optimizing = True)
    
    prod_total = jnp.sum(prod_o_hist) * dt
    
    return prod_total