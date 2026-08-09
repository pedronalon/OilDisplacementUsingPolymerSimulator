import matplotlib.pyplot as plt
import numpy as np
import json

input_path = '/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/inputs/impes_input.json'
with open(input_path, 'r') as p:
    parameters = json.load(p)

times = [50.0, 100.0, 122.24, 150.0]
t_init = parameters.get("t_init")

def impes_plot(tp):
    if tp == 'numpy' :
        path = "/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/outputs/output_numpy.txt"
        data = np.loadtxt(path,skiprows=1)
        
        path_prod = "/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/outputs/prod_output_numpy.txt"
        prod_data = np.loadtxt(path_prod,skiprows=1)
    
    elif tp == 'jax' :
        path = "/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/outputs/output_jax.txt"
        data = np.loadtxt(path,skiprows=1)
        
        path_prod = "/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/outputs/prod_output_jax.txt"
        prod_data = np.loadtxt(path_prod,skiprows=1)

    x = data[:,1]
    Sw_list = [data[:,2],data[:,4],data[:,6],data[:,8]]
    p_list = [data[:,3],data[:,5],data[:,7],data[:,9]] 
    c_list = [data[:,10],data[:,11],data[:,12],data[:,13]]

    n = len(Sw_list)
    
    plt.figure(figsize=(12,8))
    for i in range(n):
        plt.plot(x,Sw_list[i],label = 'Sw')
    
    plt.title("Water Saturation")
    plt.xlim(0.0,2000.0)
    plt.ylim(0.0,1.0)
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,8))
    for i in range(n):
        plt.plot(x,p_list[i],label = 'P')
    
    plt.title("Pressure")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12,8))
    for i in range(n):
        plt.plot(x,c_list[i],label = 'Cp')
    
    plt.title("Polymer Concentration")
    plt.grid(True)
    plt.legend()
    plt.show()

    t = prod_data[:,1]
    # prod_list = [prod_data[:,2], prod_data[:,3], prod_data[:,4], prod_data[:,5], prod_data[:,6]]

    plt.figure(figsize=(12,8))
    for i in range(len(times)):
        prod_plot = prod_data[:,i+2]
        
        plt.plot(t, prod_plot, label=f'inj time : {times[i] - t_init}')
    
    plt.title("Oil Acumulated Production")
    plt.grid(True)
    plt.legend()
    plt.show()