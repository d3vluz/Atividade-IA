import time
import math
from typing import Dict, List, Type, Any
from data_loader import Dataset, get_class_labels, k_fold_cross_validation, undersample_dataset, oversample_dataset
from algorithms import BaseClassifier
from metrics import calculate_metrics

class Evaluator:
    def __init__(self, dataset: Dataset, k: int = 10):
        self.dataset = dataset
        self.k = k
        self.labels = get_class_labels(dataset)

    def run(self, clf_class: Type[BaseClassifier], params: Dict[str, Any], sampling_strategy: str = None) -> Dict[str, List[float]]:     
        splits = k_fold_cross_validation(self.dataset, self.k)
        results = {'acc': [], 'f1': [], 'time_train': [], 'time_test': []}
        
        for fold_idx, (train, test) in enumerate(splits):
            
            if sampling_strategy == 'undersampling':
                train_processed = undersample_dataset(train, seed=42 + fold_idx)
            elif sampling_strategy == 'oversampling':
                train_processed = oversample_dataset(train, seed=42 + fold_idx)
            else:
                train_processed = train
            
            clf = clf_class(**params)

            t0 = time.time()
            clf.fit(train_processed)
            results['time_train'].append(time.time() - t0)

            t0 = time.time()
            preds = clf.predict_set(test)
            results['time_test'].append(time.time() - t0)

            y_true = [i.label for i in test]
            acc, _, f1 = calculate_metrics(y_true, preds, self.labels)
            results['acc'].append(acc)
            results['f1'].append(f1)
            
        return results

def get_stats(values: List[float], decimals: int = 4) -> str:
    """Retorna média ± desvio padrão formatado."""
    mean = sum(values) / len(values)
    var = sum((x - mean)**2 for x in values) / (len(values) - 1) if len(values) > 1 else 0
    std = math.sqrt(var)
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

def get_mean(values: List[float]) -> float:
    return sum(values) / len(values)