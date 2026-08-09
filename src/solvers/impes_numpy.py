import numpy as np
import sympy as sp 
import pandas as pd 
from scipy.sparse import diags
from scipy.special import expit



def numpy_solver(parameters):

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


    cfl = parameters.get("cfl")
    Csf = parameters.get("Csf")
    max_rrf = parameters.get("max_rrf")

    t_init = parameters.get("t_init")
    t_inj = parameters.get("t_inj")


    concentration = np.array([0.0 , 100.0 , 300.0 , 500.0])
    viscosity = np.array([0.5 , 1.5 , 4.0 , 7.8])

    fw_values = np.zeros(len(concentration))

    
    def max_fw():
        Sw = sp.symbols('Sw')

        lmbd_w = (k*(krw0 * ((Sw - Swc) /(1 - Swc - Sor))**nw)) / mu_w
        lmbd_o = (kro0 * (1 - ((Sw - Swc) / (1 - Swc - Sor)))**no) / mu_o

        fw = lmbd_w / (lmbd_w + lmbd_o)
        
        derivate_1 = sp.diff(fw,Sw)
        derivate_2 = sp.diff(derivate_1,Sw)

        maximum_point = sp.nsolve(derivate_2,Sw, [(Swc+1e-5),(1-Sor-1e-5)] , solver = 'bisect', verify = False)
        maximum_value = derivate_1.subs(Sw,maximum_point)

        return float(maximum_value)

    def Krw(Sw):
        sn = np.clip( (Sw - Swc) /(1 - Swc - Sor) , 0.0, 1.0)
        kr = krw0 * (sn)**nw
        return kr

    def Kro(Sw):
        np.clip(Sw,Sor,1-Swc)
        sn = np.clip( (Sw - Swc) /(1 - Swc - Sor) , 0.0, 1.0)
        kr = kro0 * (1 - sn)**no 
        return kr

    def linear_interpolation(c,v,x):
        return np.interp(x,c,v)

    fw_values[:] = max_fw()
    fw_max_value = np.nanmax(fw_values)
    
    prod_history = []




    for m in M:
        
        sw_history = []
        p_history = []
        c_history = []
        

        dx = (Lf-Li)/(m-1)
        v = q/a 
        vp = phi*a*dx
        # dt = cfl*(dx*phi)/(v*fw_max_value)
        dt = 0.0800827936386817
        N = int(tf/dt)
    
        prod = np.zeros(N)
        Sw = np.zeros(m)
        p = np.zeros(m)
        C = np.zeros(m)
        T = np.zeros(m)
        lmbd = np.zeros(m)
        qt = np.zeros(m-1)
        qw = np.zeros(m-1)
        T_inter = np.zeros(m-1)
        T_matrix = np.zeros((m-1,m-1))


        t = np.linspace(ti,tf,N)
        x = np.linspace(Li,Lf,m)


        Sw[:] = Swc+0.001
        p_right = 0.0

        plot_values = [int(N/4),int(N/2),int(3*N/4),N-1] 
        sw_history.append(Sw.copy())
        c_history.append(C.copy())

        for i in range (N-1):

            lmbd_w = (k*Krw(Sw))/(linear_interpolation(concentration, viscosity, C))
            lmbd_o = (k*Kro(Sw))/ mu_o 

            lmbd[:] = lmbd_o + lmbd_w

            fw_up = lmbd_w/(lmbd_o+lmbd_w)

            T[:] = (6.3283e-3 *lmbd[ :]*a)/dx
            
            main_diag = np.zeros(m-1)
            lower_diag = np.zeros(m-2)
            upper_diag = np.zeros(m-2)

            T_right = np.zeros(m-1)
            T_left = np.zeros(m-1)
            
            T_right[:] = (2 * T[:-1] * T[1:]) / (T[:-1] + T[1:])
            T_left[1:]  = (2 * T[:-2] * T[1:-1]) / (T[:-2] + T[1:-1]) 
        
            T_inter[:] = T_right
        
            #if j == 0
            main_diag[0] = T_right[0]
            upper_diag[0] = -T_right[0]
            qt[0] = q

            #if j == M-2 
            main_diag[-1] = T_left[-1] + T_right[-1]
            lower_diag[-1] = -T_left[-1]
            qt[-1] = T_right[-1] * p_right 

            # internal points 
            main_diag[1:-1] = T_left[1:-1] + T_right[1:-1]
            lower_diag[:-1] = -T_left[1:-1]
            upper_diag[1:]  = -T_right[1:-1]
            qt[1:-1] = 0.0

            T_matrix = diags([lower_diag, main_diag , upper_diag], [-1, 0, 1]).toarray()
            p[:-1] = np.linalg.solve(T_matrix,qt[:]) 
            p[-1] = p_right
        
            Q_t = -T_inter[:]*(p[1:] - p[:-1])
            
            
            qw[:] = fw_up[:-1] * Q_t
            
            Sw_old = Sw.copy()
            
            Sw[0] = Sw[0] + (dt / vp) * (q - qw[0])
            Sw[1:-1] = Sw[1:-1] + (dt / vp) * (qw[:-1] - qw[1:])
            Sw[-1] = Sw_old[-1]

            alpha = (qw*dt)/(vp)
            alpha_inj = (q*dt)/(vp)

            # polymer_injection = np.where(t[i]>= t_init and t[i]<z,500.00,-np.inf)
            # polymer_injection = expit(polymer_injection)*1000
            
            b = 1.0
            polymer_injection = expit(b*(t_inj-t[i]))*500
                

            C[-1] = ((Sw_old[-1] - alpha[-1]) * C[-1] + alpha[-1] * C[-2]) / Sw[-1]
            C[1:-1] = ((Sw_old[1:-1] - alpha[1:]) * C[1:-1] + alpha[:-1] * C[:-2]) / Sw[1:-1]
            C[0] = ((Sw_old[0] - alpha[0]) * C[0] + alpha_inj * polymer_injection) / Sw[0]
            
            np.clip(Sw, Swc, 1.0 - Sor, out = Sw)

            if(np.isnan(Sw).any()):
                break

            if i == 0:
                p_history.append(p.copy())
                

            if i+1 in plot_values:
                
                sw_history.append(Sw.copy())  
                p_history.append(p.copy())
                c_history.append(C.copy())
            
            if m == M[-1]:
                prod[i] = Q_t[-1] - qw[-1]
        
        if m == M[-1]:
            prod_history.append(np.cumsum(prod)*dt)
            
            
            
        data = {
        'x' : x , 
        
        # 'Sw_0 ' : sw_history[0],
        # 'p_0 ' : p_history[0],
        
        
        'Sw_1 ' : sw_history[1],
        'p_1 ' : p_history[1],
        
        
        'Sw_2 ' : sw_history[2],
        'p_2 ' : p_history[2],
        
        
        'Sw_3 ' : sw_history[3],
        'p_3 ' : p_history[3],
        

        'Sw_4 ' : sw_history[4],
        'p_4 ' : p_history[4],
        
        
        'C_1 ' : c_history[1],
        'C_2 ' : c_history[2],
        'C_3 ' : c_history[3],
        'C_4 ' : c_history[4],

    }  
    
    prod_data =  {

        't' : t ,
        'prod_1' : prod_history[0],

    }

    df = pd.DataFrame(data)
    prod_df = pd.DataFrame(prod_data)

     
    return df, prod_df

def calc_prod(tinj, parameters):

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

    t_init = parameters.get("t_init")

    M_0 = M[0]


    def Krw(Sw):
        sn = np.clip( (Sw - Swc) /(1 - Swc - Sor) , 0.0, 1.0)
        kr = krw0 * (sn)**nw
        return kr

    def Kro(Sw):
        np.clip(Sw,Sor,1-Swc)
        sn = np.clip( (Sw - Swc) /(1 - Swc - Sor) , 0.0, 1.0)
        kr = kro0 * (1 - sn)**no 
        return kr
    
    def linear_interpolation(c,v,x):
        return np.interp(x,c,v)

    
    concentration = np.array([0.0, 100.0, 300.0, 500.0])
    viscosity = np.array([0.5, 1.5, 4.0, 7.8])


    dx = (Lf - Li) / (M_0 - 1)
    vp = phi * a * dx
    dt = 0.0800827936386817
    N = int(tf / dt)

    Sw = np.full(M_0, Swc + 0.001)
    p = np.zeros(M_0)
    C = np.zeros(M_0)
    T = np.zeros(M_0)
    lmbd = np.zeros(M_0)
    qt = np.zeros(M_0 - 1)
    qw = np.zeros(M_0 - 1)
    T_inter = np.zeros(M_0 - 1)
    
    p_right = 0.0
    prod = 0.0
    t = np.linspace(ti, tf, N)



    for i in range(N - 1):
        lmbd_w = (k * Krw(Sw)) / (linear_interpolation(concentration, viscosity, C))
        lmbd_o = (k * Kro(Sw)) / mu_o 
        lmbd[:] = lmbd_o + lmbd_w
        fw_up = lmbd_w/(lmbd_o + lmbd_w)

        T[:] = (6.3283e-3*lmbd[:]*a)/dx
        
        main_diag = np.zeros(M_0-1)
        lower_diag = np.zeros(M_0-2)
        upper_diag = np.zeros(M_0-2)
        
        T_right = np.zeros(M_0-1)
        T_left = np.zeros(M_0-1)
        
        T_right[:] = (2 * T[:-1] * T[1:]) / (T[:-1] + T[1:])
        T_left[1:]  = (2 * T[:-2] * T[1:-1]) / (T[:-2] + T[1:-1]) 
    
        T_inter[:] = T_right
    
        #if j == 0
        main_diag[0] = T_right[0]
        upper_diag[0] = -T_right[0]
        qt[0] = q

        #if j == M-2 
        main_diag[-1] = T_left[-1] + T_right[-1]
        lower_diag[-1] = -T_left[-1]
        qt[-1] = T_right[-1] * p_right 

        # internal points 
        main_diag[1:-1] = T_left[1:-1] + T_right[1:-1]
        lower_diag[:-1] = -T_left[1:-1]
        upper_diag[1:]  = -T_right[1:-1]
        qt[1:-1] = 0.0

        T_matrix = diags([lower_diag, main_diag , upper_diag], [-1, 0, 1]).toarray()
        p[:-1] = np.linalg.solve(T_matrix,qt[:]) 
        p[-1] = p_right
    
        Q_t = -T_inter[:]*(p[1:] - p[:-1])
        
        
        qw[:] = fw_up[:-1] * Q_t
        
        Sw_old = Sw.copy()
        
        Sw[0] = Sw[0] + (dt / vp) * (q - qw[0])
        Sw[1:-1] = Sw[1:-1] + (dt / vp) * (qw[:-1] - qw[1:])
        Sw[-1] = Sw_old[-1]

        alpha = (qw*dt)/(vp)
        alpha_inj = (q*dt)/(vp)

        # polymer_injection = np.where(t[i]>= t_init and t[i]<z,500.00,-np.inf)
        # polymer_injection = expit(polymer_injection)*1000
        
        b = 1.0
        polymer_injection = expit(b*(tinj-t[i]))*500
            

        C[-1] = ((Sw_old[-1] - alpha[-1]) * C[-1] + alpha[-1] * C[-2]) / Sw[-1]
        C[1:-1] = ((Sw_old[1:-1] - alpha[1:]) * C[1:-1] + alpha[:-1] * C[:-2]) / Sw[1:-1]
        C[0] = ((Sw_old[0] - alpha[0]) * C[0] + alpha_inj * polymer_injection) / Sw[0]
        
        np.clip(Sw, Swc, 1.0 - Sor, out = Sw)

        prod += (Q_t[-1]-qw[-1])*dt
    
    return prod






