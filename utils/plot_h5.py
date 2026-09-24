import h5py
import numpy as np
import json
import matplotlib.pyplot as plt

def calcular_integral_massa(h5_path, json_path='inputs/impes_input.json'):
    # Lê os parâmetros da rocha para o cálculo correto
    with open(json_path, 'r') as r:
        params = json.load(r)
    
    phi = params.get("phi")
    a = params.get("a")

    with h5py.File(h5_path, 'r') as f:
        t_save = f['spatial/t_save'][:]
        x = f['spatial/x_centers'][:]
        
        # Pega automaticamente o último passo de tempo da simulação
        idx = len(t_save) - 1
        real_time = t_save[idx]
        
        C = f[f'spatial/C_{idx}'][:]
        Sw = f[f'spatial/Sw_{idx}'][:]
        
        funcao_massa = C * Sw * phi * a
        
        # --- MUDANÇA: REGRA DO RETÂNGULO (VOLUMES FINITOS) ---
        
        massa_fluido = np.trapezoid(funcao_massa,x)
        
        # SAÍDA EXATAMENTE COMO SOLICITADA
        print(f"mass: {massa_fluido:.4f}")

        # --- PLOTS ---
        # 1. Plot Temporal (Produção Acumulada)
        if 'temporal/prod' in f:
            plt.figure(figsize=(10, 6))
            t = f['temporal/t'][:]
            prod = f['temporal/prod'][:]
            plt.plot(t, prod, label="Produção")
            plt.title("Produção Acumulada de Óleo")
            plt.xlabel("Tempo (dias)")
            plt.ylabel("Produção Acumulada (ft³)")
            plt.grid(True)
            plt.legend()
            plt.show()

        # 2. Plots Espaciais
        nomes_espaciais = {'Sw': 'Saturação de Água', 'C': 'Concentração de Polímero', 'p': 'Pressão'}
        for var in ['Sw', 'C', 'p']:
            plt.figure(figsize=(10, 6))
            y = f[f'spatial/{var}_{idx}'][:]
            
            label_plot = f"t = {real_time:.1f} d"
            if var == 'C':
                label_plot += f" | Massa = {massa_fluido:.2f}"
                
            plt.plot(x, y, label=label_plot)
            plt.title(f"Perfil de {nomes_espaciais[var]}")
            plt.xlabel("Distância x (ft)")
            plt.ylabel(var)
            plt.grid(True)
            plt.legend()
            plt.show()

if __name__ == "__main__":
    arquivo_teste = '/home/pedro/Área de trabalho/Faculdade /OilDisplacementUsingPolymerSimulator/outputs/simulacao_tinj_122.0.h5' 
    
    try:
        calcular_integral_massa(arquivo_teste)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{arquivo_teste}' não encontrado.")
        print("Altere a variável 'arquivo_teste' no final do arquivo plot_h5.py para o nome correto.")