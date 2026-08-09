import json
import time
import pandas as pd
from src.solvers.impes_numpy import numpy_solver
from src.solvers.impes_jax import jax_solver
from utils.plots import impes_plot

def main(): 
    start = time.perf_counter()

    path = '/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/inputs/impes_input.json'
    mode = 'jax'

    with open(path, 'r') as r:
        parameters = json.load(r)
    
    times = [50.0, 100.0, 122.24, 150.0]
    
    main_prod_df = pd.DataFrame()
    
    cont = 1

    for t in times:

        parameters["t_inj"] = t
        
        if mode == 'numpy':
            df, prod_df = numpy_solver(parameters)
            output = '/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/outputs/output_numpy.txt'
            prod_output = '/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/outputs/prod_output_numpy.txt'
        elif mode == 'jax':
            df, prod_df = jax_solver(parameters)
            output = '/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/outputs/output_jax.txt'
            prod_output = '/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/outputs/prod_output_jax.txt'
        else:
            return
        
        with open(output, 'w') as archive:
            archive.write(df.to_string())
        
        if cont == 1:
            main_prod_df['t'] = prod_df['t'] 
            
        main_prod_df[f'prod_{cont}'] = prod_df['prod_1'] 
        
        with open(prod_output, 'w') as archive:
            archive.write(main_prod_df.to_string())
            
        cont += 1



    end = time.perf_counter()
    print('\nTime: ', end - start)
    
    impes_plot(mode)



main()