import h5py
import matplotlib.pyplot as plt
import os
import glob

def plot_producao_multipla(pasta_outputs):
    
    arquivos = glob.glob(os.path.join(pasta_outputs, '*.h5'))
    
    arquivos.sort() 
    
    if not arquivos:
        print(f"Nenhum arquivo .h5 encontrado na pasta {pasta_outputs}/.")
        return

    plt.figure(figsize=(12, 8))
    
    for file in arquivos:
        nome_base = os.path.basename(file)
        
        if 'tinj_' in nome_base:
            valor_tinj = nome_base.replace('.h5', '').split('tinj_')[-1]
            label = f"t_inj = {valor_tinj} dias"
        else:
            label = nome_base
            
        with h5py.File(file, 'r') as f:
            # Verifica se a produção foi salva neste arquivo
            if 'temporal/prod' in f:
                t = f['temporal/t'][:]
                prod = f['temporal/prod'][:]
                
                # Plota a linha deste T_inj no gráfico principal
                plt.plot(t, prod, linewidth=2, label=label)
                
    plt.title("Produção Acumulada de Óleo")
    plt.xlabel("Tempo de Simulação (dias)")
    plt.ylabel("Produção Acumulada de Óleo (ft³)")
    plt.grid(True)

    plt.ylim(0.0,500000.00)
    plt.xlim(0.0,800.00)
    plt.legend() 
    plt.show()

if __name__ == "__main__":
    # Define a pasta onde estão os arquivos da simulação
    pasta = 'outputs'
    plot_producao_multipla(pasta)