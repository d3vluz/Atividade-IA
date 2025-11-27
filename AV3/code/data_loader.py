import os
import random
from typing import List, Tuple, Any

class Instance:
    """Representa uma única instância (amostra) do dataset."""
    def __init__(self, features: List[Any], label: Any):
        self.features = features
        self.label = label

Dataset = List[Instance]

def _parse_arff_line(line: str, attributes: List[Tuple[str, Any]], class_index: int) -> Instance:
    """Helper para processar uma linha de dados."""
    values = [v.strip() for v in line.split(',')]
    if len(values) != len(attributes):
        return None
    
    features = []
    label = None
    
    for i, (_, attr_type) in enumerate(attributes):
        val = values[i]
        try:
            parsed = int(val) 
        except ValueError:
            parsed = val
            
        if i == class_index:
            label = parsed
        else:
            features.append(parsed)
            
    return Instance(features, label)

def load_dataset_from_file(file_path: str) -> Tuple[Dataset, List[str]]:
    """Lê o arquivo ARFF do disco e retorna o dataset estruturado."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    dataset: Dataset = []
    attributes: List[Tuple[str, str]] = []
    data_section_found = False

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'): continue

            if line.lower().startswith('@attribute'):
                parts = line.split()
                name = parts[1].strip("'")
                attr_type = ' '.join(parts[2:])
                attributes.append((name, attr_type))
            
            elif line.lower().startswith('@data'):
                data_section_found = True
                continue
            
            elif data_section_found:
                inst = _parse_arff_line(line, attributes, len(attributes) - 1)
                if inst:
                    dataset.append(inst)

    return dataset, [attr[0] for attr in attributes]

def get_class_labels(dataset: Dataset) -> List[Any]:
    return sorted(list(set(inst.label for inst in dataset)))

def undersample_dataset(dataset: Dataset, seed: int = None) -> Dataset:
    """Aplica undersampling aleatório (corta classes maiores)."""
    if seed is not None:
        random.seed(seed)
    
    grouped = {}
    for inst in dataset:
        grouped.setdefault(inst.label, []).append(inst)
    
    min_size = min(len(instances) for instances in grouped.values())
    
    balanced = []
    for label, instances in grouped.items():
        sampled = random.sample(instances, min_size)
        balanced.extend(sampled)
    
    random.shuffle(balanced)
    return balanced

def oversample_dataset(dataset: Dataset, seed: int = None) -> Dataset:
    """Aplica Random Oversampling (duplica classes menores)."""
    if seed is not None:
        random.seed(seed)
    
    grouped = {}
    for inst in dataset:
        grouped.setdefault(inst.label, []).append(inst)
    
    max_size = max(len(instances) for instances in grouped.values())
    
    balanced = []
    for label, instances in grouped.items():
        balanced.extend(instances) # Mantém originais
        
        n_missing = max_size - len(instances)
        if n_missing > 0:
            # Sorteia com reposição
            extras = [random.choice(instances) for _ in range(n_missing)]
            balanced.extend(extras)
    
    random.shuffle(balanced)
    return balanced

def k_fold_cross_validation(dataset: Dataset, k: int = 10) -> List[Tuple[Dataset, Dataset]]:
    """Gera os splits de treino e teste com estratificação (Stratified K-Fold)."""
    grouped = {}
    for inst in dataset:
        grouped.setdefault(inst.label, []).append(inst)
    
    folds = [[] for _ in range(k)]
    
    for label, instances in grouped.items():
        fold_size = len(instances) // k
        
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(instances)
            folds[i].extend(instances[start:end])

    splits = []
    for i in range(k):
        test_set = folds[i]
        train_set = [inst for j in range(k) if j != i for inst in folds[j]]
        splits.append((train_set, test_set))
        
    return splits