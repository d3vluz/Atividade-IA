from typing import List, Any, Dict, Tuple

def calculate_confusion_matrix(y_true: List[Any], y_pred: List[Any], labels: List[Any]) -> Dict[Any, Dict[str, int]]:
    """Cálculo da Matriz de confusão por classe."""
    matrix = {label: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for label in labels}
    
    for t, p in zip(y_true, y_pred):
        if t == p:
            matrix[t]['TP'] += 1
            for label in labels:
                if label != t: matrix[label]['TN'] += 1
        else:
            matrix[t]['FN'] += 1
            matrix[p]['FP'] += 1
            for label in labels:
                if label != t and label != p: matrix[label]['TN'] += 1
    return matrix

def calculate_metrics(y_true: List[Any], y_pred: List[Any], labels: List[Any]) -> Tuple[float, float, float]:
    """Retorna (Accuracy, Macro-Precision, Macro-F1)."""
    if not y_true: return 0.0, 0.0, 0.0
    
    # Acurácia Global
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)
    
    # Macro Metrics
    matrix = calculate_confusion_matrix(y_true, y_pred, labels)
    precisions = []
    f1s = []
    
    for label in labels:
        tp = matrix[label]['TP']
        fp = matrix[label]['FP']
        fn = matrix[label]['FN']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        f1s.append(f1)
    
    macro_precision = sum(precisions) / len(labels)
    macro_f1 = sum(f1s) / len(labels)
    
    return accuracy, macro_precision, macro_f1