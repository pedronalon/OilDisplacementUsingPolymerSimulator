import sys
import json
import time
import argparse
from src.solvers.impes_jax import jax_solver

def main(): 
    # --- 1. CONFIGURAÇÃO DA LINHA DE COMANDO ---
    parser = argparse.ArgumentParser(description="Simulador de Polímero IMPES JAX")
    parser.add_argument('vars', type=str, help="Variáveis para plotar ex: Sw,C,p,prod")
    parser.add_argument('dt', type=float, help="Valor do dt_save em dias ex: 1.0")
    
    # Argumento opcional para facilitar a variação de parâmetros no futuro
    parser.add_argument('--t_inj', type=float, default=None, help="Tempo de injeção (sobrescreve o JSON se fornecido)")

    args = parser.parse_args()

    # Formata a lista de variáveis
    variables_to_plot = [v.strip() for v in args.vars.split(',')]
    dt_save = args.dt

    # --- 2. LEITURA DE PARÂMETROS ---
    start = time.perf_counter()
    path = 'inputs/impes_input.json'
    
    with open(path, 'r') as r:
        parameters = json.load(r)
    
    # Se você digitou --t_inj no terminal, ele substitui o valor do JSON
    if args.t_inj is not None:
        parameters["t_inj"] = args.t_inj
        
    t_val = parameters.get("t_inj")
    prefix = f"simulacao_tinj_{t_val}"
    
    print(f'dt_save = {dt_save}')
    
    # --- 3. EXECUÇÃO DA SIMULAÇÃO ---
    h5_path = jax_solver(parameters, dt_save=dt_save, file_prefix=prefix)

    end = time.perf_counter()
    print(f'execution time: {end - start:.2f}\n')
    
    # # --- 4. CHAMADA AUTOMÁTICA DOS GRÁFICOS ---
    # tempo_final = parameters.get("tf", 800.0) 
    
    # print(f"Plots for: {variables_to_plot}")
    # # O plot_results aceita listas, então passamos o arquivo único dentro de colchetes
    # plot_results([h5_path], [f"T inj = {t_val}"], variables=variables_to_plot, target_time=tempo_final)

if __name__ == "__main__":
    main()


















# import json
# import time
# import pandas as pd
# from src.solvers.impes_numpy import numpy_solver
# from src.solvers.impes_jax import jax_solver
# from utils.plots import impes_plot
# from flax.core import FrozenDict
# def main(): 
#     start = time.perf_counter()

#     path = '/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/inputs/impes_input.json'
#     mode = 'jax'

#     with open(path, 'r') as r:
#         parameters = json.load(r)
    
#     times = [50.0, 100.0, 122.24, 150]
    
#     main_prod_df = pd.DataFrame()
    
#     cont = 1

#     for t in times:

#         parameters["t_inj"] = t
        
#         if mode == 'numpy':
#             df, prod_df = numpy_solver(parameters)
#             output = '/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/outputs/output_numpy.txt'
#             prod_output = '/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/outputs/prod_output_numpy.txt'
#         elif mode == 'jax':
#             df, prod_df = jax_solver(parameters)
#             output = '/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/outputs/output_jax.txt'
#             prod_output = '/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/outputs/prod_output_jax.txt'
#         else:
#             return
        
#         with open(output, 'w') as archive:
#             archive.write(df.to_string())
        
#         if cont == 1:
#             main_prod_df['t'] = prod_df['t']
#             main_prod_df['pvi'] = prod_df['pvi'] 
            
#         main_prod_df[f'prod_{cont}'] = prod_df['prod_1'] 
#         main_prod_df[f'C_{cont}'] = prod_df["Cp"]
        
#         with open(prod_output, 'w') as archive:
#             archive.write(main_prod_df.to_string())
            
#         cont += 1



#     end = time.perf_counter()
#     print('\nTime: ', end - start)
    
#     impes_plot(mode)



# main()