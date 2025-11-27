import os
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_dataset_from_file
from algorithms import KNNClassifier, PerceptronClassifier, GaussianNaiveBayes, MultivariateNaiveBayes, MLPClassifier, euclidean_distance, manhattan_distance
from use_case import Evaluator, get_stats, get_mean

script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(script_dir, '..', 'data', 'dataset_23.arff') 
FILE_PATH = os.path.normpath(FILE_PATH)

RESULTS_DIR = os.path.join(script_dir,'..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def print_header(title):
    print("\n" + "="*130)
    print(title.center(130))
    print("="*130)
    print(f"{'Classificador':<30} │ {'Acurácia':^20} │ {'F1-Score':^20} │ {'Treino (ms)':^22} │ {'Teste (ms)':^22}")
    print("─" * 130)

def plot_algorithm_dashboard(algo_name, strategies, metrics):
    """
    Gera um painel com 2 gráficos comparativos:
    1. Qualidade (Acurácia vs F1)
    2. Tempo (Treino vs Teste)
    Salva na pasta 'results/'.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Análise Completa: {algo_name}', fontsize=16)
    
    x = np.arange(len(strategies))
    width = 0.35

    rects1 = ax1.bar(x - width/2, metrics['acc'], width, label='Acurácia', color='#4c72b0')
    rects2 = ax1.bar(x + width/2, metrics['f1'], width, label='F1', color='#dd8452')
    
    ax1.set_title('Qualidade do Modelo')
    ax1.set_ylabel('Score (0-1)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    rects3 = ax2.bar(x - width/2, metrics['time_train'], width, label='Treino (ms)', color='#55a868')
    rects4 = ax2.bar(x + width/2, metrics['time_test'], width, label='Teste (ms)', color='#c44e52')

    ax2.set_title('Performance Computacional (Milissegundos)')
    ax2.set_ylabel('Tempo (ms)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies)
    max_height = max(max(metrics['time_train']), max(metrics['time_test']))
    ax2.set_ylim(0, max_height * 1.15) 
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(ax, rects, fmt='{:.2f}'):
        if not isinstance(rects, list) and not hasattr(rects, '__iter__'): rects = [rects]
        if hasattr(rects, 'patches'): rects = rects.patches
        for rect in rects:
            height = rect.get_height()
            ax.annotate(fmt.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(ax1, rects1)
    autolabel(ax1, rects2)
    autolabel(ax2, rects3)
    autolabel(ax2, rects4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f"dashboard_{algo_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    # print(f"   [Dashboard salvo: {save_path}]")

def main():
    try:
        dataset, attrs = load_dataset_from_file(FILE_PATH)
        print(f"Dataset carregado: {len(dataset)} instâncias.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {FILE_PATH}")
        return

    configs = [
        ("KNN (Euclidiana)", KNNClassifier, {'k': 5, 'distance_func': euclidean_distance}),
        ("KNN (Manhattan)", KNNClassifier, {'k': 5, 'distance_func': manhattan_distance}),
        ("Naive Bayes (Univariada)", GaussianNaiveBayes, {}),
        ("Naive Bayes (Multivariada)", MultivariateNaiveBayes, {}),
        ("Perceptron", PerceptronClassifier, {'learning_rate': 0.01, 'epochs': 1000}),
        ("MLP (10 Neurônios)", MLPClassifier, {'n_hidden': 10, 'learning_rate': 0.1, 'epochs': 100}),
    ]

    scenarios = [
        ("NORMAL (Sem Balanceamento)", None),
        ("UNDERSAMPLING", "undersampling"),
        ("OVERSAMPLING", "oversampling")
    ]

    evaluator = Evaluator(dataset, k=10)
    
    plots_data = {
        name: {
            'strategies': [], 
            'acc': [], 
            'f1': [], 
            'time_train': [], 
            'time_test': []
        } for name, _, _ in configs
    }
    
    for scenario_name, strategy_code in scenarios:
        print_header(f"RESULTADOS: {scenario_name}")
        
        for name, clf_cls, params in configs:
            res = evaluator.run(clf_cls, params, sampling_strategy=strategy_code)
            times_train_ms = [t * 1000 for t in res['time_train']]
            times_test_ms = [t * 1000 for t in res['time_test']]
            
            row_str = f"{name:<30} │ {get_stats(res['acc'], 3):^20} │ {get_stats(res['f1'], 3):^20} │ {get_stats(times_train_ms, 2):^22} │ {get_stats(times_test_ms, 2):^22}"
            print(row_str)
            
            strategy_label = scenario_name.split()[0]
            plots_data[name]['strategies'].append(strategy_label)
            plots_data[name]['acc'].append(get_mean(res['acc']))
            plots_data[name]['f1'].append(get_mean(res['f1']))
            plots_data[name]['time_train'].append(get_mean(times_train_ms))
            plots_data[name]['time_test'].append(get_mean(times_test_ms))
            
        print("─" * 130)

    # print(f"\n📊 Gerando Dashboards na pasta '{RESULTS_DIR}'...")
    for name, data in plots_data.items():
        plot_algorithm_dashboard(name, data['strategies'], data)
        
    print("\nConcluído! Verifique a pasta 'results' para ver os gráficos.")

if __name__ == "__main__":
    main()