import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_evolucao_espacial(arquivo, variavel, tempos_alvo):
    """Plota a evolução espacial de uma variável em múltiplos instantes de tempo."""
    nomes_espaciais = {'Sw': 'Saturação da Água', 'C': 'Concentração de Polímero', 'p': 'Pressão'}
    
    plt.figure(figsize=(12, 8))
    
    with h5py.File(arquivo, 'r') as f:
        x = f['spatial/x_centers'][:]
        t_save = f['spatial/t_save'][:]
        
        # Para cada tempo desejado na lista, encontra a linha correspondente na matriz e plota
        for target_time in tempos_alvo:
            idx = (np.abs(t_save - target_time)).argmin()
            real_time = t_save[idx]
            
            # Fatiando a matriz 2D (linha = tempo, colunas = espaço)
            y = f[f'spatial/{variavel}'][idx, :]
            
            # O matplotlib cuida automaticamente de mudar as cores das linhas
            plt.plot(x, y, linewidth=2, label=f"t = {real_time:.1f} dias")
            
    plt.title(f"{nomes_espaciais[variavel]}")
    plt.xlabel("Distância x (ft)")
    plt.ylabel(variavel)
    plt.xlim(0.0,2000.0)
    plt.ylim(0.0,1.0)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Escolha o arquivo da simulação que deseja analisar
    arquivo = 'outputs/simulacao_tinj_0.0.h5'
    
    # Escolha a variável que deseja ver: 'Sw', 'C' ou 'p'
    variavel_analise = 'Sw'
    
    # Adicione aqui todos os dias que deseja comparar no gráfico
    tempos_para_plotar = [0.0,100.00, 200.00, 400.00, 600.00, 800.00]
    
    try:
        plot_evolucao_espacial(arquivo, variavel_analise, tempos_para_plotar)
    except FileNotFoundError as e:
        print("Erro: Ficheiro não encontrado. Verifique o caminho.")