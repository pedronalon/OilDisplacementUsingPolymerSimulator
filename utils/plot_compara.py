import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

def comparar_simulacoes(file1, file2, target_time):
    
    label1 = os.path.basename(file1)
    label2 = os.path.basename(file2)
    
    plt.figure(figsize=(10, 6))
    
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        if 'temporal/prod' in f1 and 'temporal/prod' in f2:
            t1 = f1['temporal/t'][:]
            prod1 = f1['temporal/prod'][:]
            plt.plot(t1, prod1, linewidth=2, label=label1)
            
            t2 = f2['temporal/t'][:]
            prod2 = f2['temporal/prod'][:]
            plt.plot(t2, prod2, linewidth=2, linestyle='--', label=label2)
            
            plt.title("Comparação: Produção Acumulada de Óleo")
            plt.xlabel("Tempo (dias)")
            plt.ylabel("Produção Acumulada (ft³)")
            plt.grid(True)
            plt.legend()
            plt.show()

    nomes_espaciais = {'Sw': 'Saturação de Água', 'C': 'Concentração de Polímero', 'p': 'Pressão'}
    
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        x1 = f1['spatial/x_centers'][:]
        x2 = f2['spatial/x_centers'][:]
        
        t_save1 = f1['spatial/t_save'][:]
        idx1 = (np.abs(t_save1 - target_time)).argmin()
        
        
        t_save2 = f2['spatial/t_save'][:]
        idx2 = (np.abs(t_save2 - target_time)).argmin()
        

        for var in ['Sw', 'C', 'p']:
            plt.figure(figsize=(10, 6))
            
            y1 = f1[f'spatial/{var}'][idx1, :]
            y2 = f2[f'spatial/{var}'][idx2, :]
                        
            plt.plot(x1, y1, linewidth=2, label=f"{label1} | t = {t_save1[idx1]:.2f} d")
            plt.plot(x2, y2, linewidth=2, linestyle='--', label=f"{label2} | t = {t_save2[idx2]:.2f} d")
                        
            plt.title(f"Comparação de Perfil: {nomes_espaciais[var]}")
            plt.xlabel("Distância x (ft)")
            plt.ylabel(var)
            plt.grid(True)
            plt.legend()
            plt.show()

if __name__ == "__main__":
    arquivo_1 = 'outputs/simulacao_tinj_0.0.h5'
    arquivo_2 = 'outputs/simulacao_tinj_150.0.h5'
    
    tempo_analise = 952.00
    
    try:
        comparar_simulacoes(arquivo_1, arquivo_2, tempo_analise)
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Verifique se os caminhos estão corretos.")
        print(e)