import os
import json
from typing import Dict, List, Any
from data_loader import load_dataset_from_file
from algorithms import KNNClassifier, PerceptronClassifier, MLPClassifier, euclidean_distance, manhattan_distance
from use_case import Evaluator, get_mean

script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(script_dir, '..', 'data', 'dataset_23.arff')
FILE_PATH = os.path.normpath(FILE_PATH)

RESULTS_FILE = os.path.join(script_dir, '..', 'results', 'hyperparameter_results.json')


def test_hyperparameters(dataset, param_grid: Dict[str, List], clf_class, clf_name: str):
    """Testa diferentes combinações de hiperparâmetros usando validação cruzada."""
    print(f"\n{'='*80}")
    print(f"Testando hiperparâmetros para: {clf_name}")
    print(f"{'='*80}")
    
    evaluator = Evaluator(dataset, k=10)
    results = []
    
    # Gerar todas as combinações de parâmetros
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    def generate_combinations(names, values, current={}):
        if not names:
            yield dict(current)
            return
        
        name = names[0]
        for value in values[0]:
            current[name] = value
            yield from generate_combinations(names[1:], values[1:], current)
    
    combinations = list(generate_combinations(param_names, param_values))
    total = len(combinations)
    
    for idx, params in enumerate(combinations, 1):
        print(f"\n[{idx}/{total}] Testando: {params}")
        
        # Testar sem balanceamento (NORMAL)
        res = evaluator.run(clf_class, params, sampling_strategy=None)
        
        avg_acc = get_mean(res['acc'])
        avg_f1 = get_mean(res['f1'])
        avg_train_time = get_mean(res['time_train']) * 1000  # ms
        avg_test_time = get_mean(res['time_test']) * 1000    # ms
        
        result = {
            'params': params,
            'accuracy': avg_acc,
            'f1_score': avg_f1,
            'train_time_ms': avg_train_time,
            'test_time_ms': avg_test_time
        }
        results.append(result)
        
        print(f"  → Acurácia: {avg_acc:.4f} | F1: {avg_f1:.4f} | "
              f"Treino: {avg_train_time:.2f}ms | Teste: {avg_test_time:.2f}ms")
    
    # Ordenar por F1-Score (melhor métrica para dados desbalanceados)
    results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"TOP 5 MELHORES CONFIGURAÇÕES (ordenado por F1-Score):")
    print(f"{'='*80}")
    
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. Parâmetros: {result['params']}")
        print(f"   Acurácia: {result['accuracy']:.4f}")
        print(f"   F1-Score: {result['f1_score']:.4f}")
        print(f"   Tempo Treino: {result['train_time_ms']:.2f}ms")
        print(f"   Tempo Teste: {result['test_time_ms']:.2f}ms")
    
    return {
        'algorithm': clf_name,
        'all_results': results,
        'best_params': results[0]['params'],
        'best_f1_score': results[0]['f1_score'],
        'best_accuracy': results[0]['accuracy']
    }


def main():
    print("="*80)
    print("OTIMIZAÇÃO DE HIPERPARÂMETROS")
    print("="*80)
    
    try:
        dataset, _ = load_dataset_from_file(FILE_PATH)
        print(f"Dataset carregado: {len(dataset)} instâncias.\n")
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {FILE_PATH}")
        return
    
    all_results = {}
    
    # =========================================================================
    # KNN - Testar diferentes valores de k e funções de distância
    # =========================================================================
    print("\n" + ">"*80)
    print("1. KNN - Testando k e funções de distância")
    print(">"*80)
    
    knn_euclidean_grid = {
        'k': [3, 5, 7, 9, 11, 15],
        'distance_func': [euclidean_distance]
    }
    
    knn_manhattan_grid = {
        'k': [3, 5, 7, 9, 11, 15],
        'distance_func': [manhattan_distance]
    }
    
    knn_euc_results = test_hyperparameters(dataset, knn_euclidean_grid, KNNClassifier, "KNN (Euclidiana)")
    knn_man_results = test_hyperparameters(dataset, knn_manhattan_grid, KNNClassifier, "KNN (Manhattan)")
    
    all_results['KNN_Euclidean'] = knn_euc_results
    all_results['KNN_Manhattan'] = knn_man_results
    
    # =========================================================================
    # Perceptron - Testar learning_rate e epochs
    # =========================================================================
    print("\n" + ">"*80)
    print("2. PERCEPTRON - Testando learning_rate e epochs")
    print(">"*80)
    
    perceptron_grid = {
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'epochs': [50, 100, 500, 1000]
    }
    
    perceptron_results = test_hyperparameters(dataset, perceptron_grid, PerceptronClassifier, "Perceptron")
    all_results['Perceptron'] = perceptron_results
    
    # =========================================================================
    # MLP - Testar n_hidden, learning_rate, e epochs
    # =========================================================================
    print("\n" + ">"*80)
    print("3. MLP - Testando neurônios ocultos, learning_rate e epochs")
    print(">"*80)
    
    mlp_grid = {
        'n_hidden': [5, 10, 20, 30],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'epochs': [50, 100, 200]
    }
    
    mlp_results = test_hyperparameters(dataset, mlp_grid, MLPClassifier, "MLP")
    all_results['MLP'] = mlp_results
    
    # =========================================================================
    # Salvar resultados
    # =========================================================================
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    # Converter distance_func para string para serialização JSON
    for key in all_results:
        for result in all_results[key]['all_results']:
            if 'distance_func' in result['params']:
                func = result['params']['distance_func']
                result['params']['distance_func'] = func.__name__
        
        if 'distance_func' in all_results[key]['best_params']:
            func = all_results[key]['best_params']['distance_func']
            all_results[key]['best_params']['distance_func'] = func.__name__
    
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Resultados salvos em: {RESULTS_FILE}")
    print(f"{'='*80}")
    
    # =========================================================================
    # Resumo Final
    # =========================================================================
    print(f"\n\n{'#'*80}")
    print("RESUMO - MELHORES HIPERPARÂMETROS ENCONTRADOS")
    print(f"{'#'*80}\n")
    
    for algo_name, data in all_results.items():
        print(f"📌 {data['algorithm']}")
        print(f"   Melhores Parâmetros: {data['best_params']}")
        print(f"   F1-Score: {data['best_f1_score']:.4f}")
        print(f"   Acurácia: {data['best_accuracy']:.4f}\n")
    
    print("="*80)
    print("RECOMENDAÇÕES PARA ATUALIZAR O main.py:")
    print("="*80)
    print("\nconfigs = [")
    
    # KNN Euclidean
    knn_euc = all_results['KNN_Euclidean']
    k_euc = knn_euc['best_params']['k']
    print(f'    ("KNN (Euclidiana)", KNNClassifier, {{"k": {k_euc}, "distance_func": euclidean_distance}}),')
    
    # KNN Manhattan
    knn_man = all_results['KNN_Manhattan']
    k_man = knn_man['best_params']['k']
    print(f'    ("KNN (Manhattan)", KNNClassifier, {{"k": {k_man}, "distance_func": manhattan_distance}}),')
    
    print('    ("Naive Bayes (Univariada)", GaussianNaiveBayes, {}),')
    print('    ("Naive Bayes (Multivariada)", MultivariateNaiveBayes, {}),')
    
    # Perceptron
    perc = all_results['Perceptron']
    print(f'    ("Perceptron", PerceptronClassifier, {perc["best_params"]}),')
    
    # MLP
    mlp = all_results['MLP']
    mlp_params = mlp['best_params'].copy()
    mlp_params['seed'] = 42  # Adicionar seed para reprodutibilidade
    print(f'    ("MLP (10 Neurônios)", MLPClassifier, {mlp_params}),')
    
    print("]\n")


if __name__ == "__main__":
    main()
