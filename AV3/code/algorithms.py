import math
import random
from typing import List, Any, Dict, Callable
from data_loader import Dataset


def euclidean_distance(x1: List[Any], x2: List[Any]) -> float:
    """Calcula a distância euclidiana entre dois vetores."""
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(x1, x2)))


def manhattan_distance(x1: List[Any], x2: List[Any]) -> float:
    """Calcula a distância de Manhattan entre dois vetores."""
    return sum(abs(float(a) - float(b)) for a, b in zip(x1, x2))


class BaseClassifier:
    """Classe base abstrata para classificadores."""
    
    def fit(self, train_set: Dataset) -> None:
        raise NotImplementedError
    
    def predict(self, features: List[Any]) -> Any:
        raise NotImplementedError
    
    def predict_set(self, test_set: Dataset) -> List[Any]:
        return [self.predict(instance.features) for instance in test_set]


class KNNClassifier(BaseClassifier):
    """K-Nearest Neighbors Classifier."""
    
    def __init__(self, k: int = 5, distance_func: Callable = euclidean_distance):
        self.k = k
        self.distance_func = distance_func
        self.training_data: Dataset = []

    def fit(self, train_set: Dataset) -> None:
        self.training_data = train_set

    def predict(self, features: List[Any]) -> Any:
        distances = [
            (self.distance_func(features, item.features), item.label)
            for item in self.training_data
        ]
        distances.sort(key=lambda x: x[0])
        
        k_nearest_labels = [label for _, label in distances[:self.k]]
        return max(set(k_nearest_labels), key=k_nearest_labels.count)


class PerceptronClassifier(BaseClassifier):
    """Perceptron multiclasse usando estratégia One-vs-All."""
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 50):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: Dict[Any, List[float]] = {}
        self.bias: Dict[Any, float] = {}
        self.labels: List[Any] = []

    def _step_function(self, x: float) -> int:
        """Step function: retorna 1 se x >= 0, senão 0."""
        return 1 if x >= 0 else 0

    def _compute_activation(self, features: List[float], label: Any) -> float:
        """Calcula a saída linear (produto escalar + bias)."""
        return sum(f * self.weights[label][i] for i, f in enumerate(features)) + self.bias[label]

    def fit(self, train_set: Dataset) -> None:
        self.labels = sorted(set(instance.label for instance in train_set))
        n_features = len(train_set[0].features)
        
        for label in self.labels:
            self.weights[label] = [0.0] * n_features
            self.bias[label] = 0.0
        
        for _ in range(self.epochs):
            for instance in train_set:
                features = [float(x) for x in instance.features]
                
                for label in self.labels:
                    y_target = 1 if instance.label == label else 0
                    linear_output = self._compute_activation(features, label)
                    y_predicted = self._step_function(linear_output)
                    
                    if y_target != y_predicted:
                        update = self.learning_rate * (y_target - y_predicted)
                        for i in range(n_features):
                            self.weights[label][i] += update * features[i]
                        self.bias[label] += update

    def predict(self, features: List[Any]) -> Any:
        features_float = [float(x) for x in features]
        scores = {
            label: self._compute_activation(features_float, label)
            for label in self.labels
        }
        return max(scores, key=scores.get)


class GaussianNaiveBayes(BaseClassifier):
    """Gaussian Naive Bayes Classifier."""
    
    MIN_VARIANCE = 1e-9
    
    def __init__(self):
        self.statistics: Dict[Any, Dict[int, Dict[str, float]]] = {}
        self.priors: Dict[Any, float] = {}

    def fit(self, train_set: Dataset) -> None:
        total = len(train_set)
        grouped: Dict[Any, List] = {}
        
        for instance in train_set:
            grouped.setdefault(instance.label, []).append(instance)
        
        n_features = len(train_set[0].features)
        
        for label, instances in grouped.items():
            self.priors[label] = len(instances) / total
            self.statistics[label] = {}
            
            for i in range(n_features):
                values = [float(inst.features[i]) for inst in instances]
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                std = math.sqrt(max(variance, self.MIN_VARIANCE))
                
                self.statistics[label][i] = {'mean': mean, 'std': std}

    def _log_pdf(self, x: float, mean: float, std: float) -> float:
        """Calcula o log da PDF gaussiana (mais estável numericamente)."""
        return -0.5 * math.log(2 * math.pi) - math.log(std) - ((x - mean) ** 2) / (2 * std ** 2)

    def predict(self, features: List[Any]) -> Any:
        best_label = None
        max_log_prob = -math.inf
        
        for label, prior in self.priors.items():
            log_prob = math.log(prior)
            
            for i, value in enumerate(features):
                stats = self.statistics[label][i]
                log_prob += self._log_pdf(float(value), stats['mean'], stats['std'])
            
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_label = label
        
        return best_label


class MultivariateNaiveBayes(BaseClassifier):
    """
    Naive Bayes Multivariado (Gaussiano).
    
    Utiliza matriz de covariância completa por classe para modelar
    a distribuição gaussiana multivariada.
    """
    
    REGULARIZATION = 1e-6
    DETERMINANT_MIN = 1e-10
    
    def __init__(self):
        self.means: Dict[Any, List[float]] = {}
        self.covariance_matrices: Dict[Any, List[List[float]]] = {}
        self.priors: Dict[Any, float] = {}

    def _compute_covariance_matrix(self, data: List[List[float]], mean: List[float]) -> List[List[float]]:
        """Calcula a matriz de covariância com regularização."""
        n_samples = len(data)
        n_features = len(mean)
        
        cov = [[0.0] * n_features for _ in range(n_features)]
        
        for sample in data:
            for i in range(n_features):
                for j in range(n_features):
                    cov[i][j] += (sample[i] - mean[i]) * (sample[j] - mean[j])
        
        divisor = max(n_samples - 1, 1)
        for i in range(n_features):
            for j in range(n_features):
                cov[i][j] /= divisor
            cov[i][i] += self.REGULARIZATION
        
        return cov

    def _compute_determinant(self, matrix: List[List[float]]) -> float:
        """Calcula o determinante usando eliminação de Gauss com pivoteamento."""
        n = len(matrix)
        
        if n == 1:
            return matrix[0][0]
        if n == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        
        m = [row[:] for row in matrix]
        determinant = 1.0
        
        for i in range(n):
            max_row = max(range(i, n), key=lambda k: abs(m[k][i]))
            
            if max_row != i:
                m[i], m[max_row] = m[max_row], m[i]
                determinant *= -1
            
            if abs(m[i][i]) < self.DETERMINANT_MIN:
                return self.DETERMINANT_MIN
            
            for k in range(i + 1, n):
                factor = m[k][i] / m[i][i]
                for j in range(i, n):
                    m[k][j] -= factor * m[i][j]
        
        for i in range(n):
            determinant *= m[i][i]
        
        return determinant if abs(determinant) > self.DETERMINANT_MIN else self.DETERMINANT_MIN

    def _compute_inverse(self, matrix: List[List[float]]) -> List[List[float]]:
        """Calcula a inversa usando eliminação de Gauss-Jordan."""
        n = len(matrix)
        
        augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
        
        for i in range(n):
            max_row = max(range(i, n), key=lambda k: abs(augmented[k][i]))
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
            
            pivot = augmented[i][i]
            if abs(pivot) < self.DETERMINANT_MIN:
                pivot = self.DETERMINANT_MIN
            
            for j in range(2 * n):
                augmented[i][j] /= pivot
            
            for k in range(n):
                if k != i:
                    factor = augmented[k][i]
                    for j in range(2 * n):
                        augmented[k][j] -= factor * augmented[i][j]
        
        return [[augmented[i][j + n] for j in range(n)] for i in range(n)]

    def fit(self, train_set: Dataset) -> None:
        total = len(train_set)
        grouped: Dict[Any, List] = {}
        
        for instance in train_set:
            grouped.setdefault(instance.label, []).append(instance)
        
        for label, instances in grouped.items():
            self.priors[label] = len(instances) / total
            data = [[float(x) for x in inst.features] for inst in instances]
            n_features = len(data[0])
            
            self.means[label] = [
                sum(sample[i] for sample in data) / len(data)
                for i in range(n_features)
            ]
            self.covariance_matrices[label] = self._compute_covariance_matrix(data, self.means[label])

    def _compute_log_likelihood(self, x: List[float], mean: List[float], cov: List[List[float]]) -> float:
        """Calcula o log da PDF gaussiana multivariada."""
        d = len(x)
        diff = [x[i] - mean[i] for i in range(d)]
        
        det = self._compute_determinant(cov)
        inv_cov = self._compute_inverse(cov)
        
        mahalanobis = sum(
            diff[i] * inv_cov[i][j] * diff[j]
            for i in range(d)
            for j in range(d)
        )
        
        log_likelihood = -0.5 * (d * math.log(2 * math.pi) + math.log(det) + mahalanobis)
        return log_likelihood

    def predict(self, features: List[Any]) -> Any:
        x = [float(f) for f in features]
        best_label = None
        max_log_prob = -math.inf
        
        for label, prior in self.priors.items():
            log_prob = math.log(prior) + self._compute_log_likelihood(
                x, self.means[label], self.covariance_matrices[label]
            )
            
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_label = label
        
        return best_label


class MLPClassifier(BaseClassifier):
    """Multi-Layer Perceptron com uma camada oculta."""
    
    def __init__(self, n_hidden: int = 10, learning_rate: float = 0.1, epochs: int = 100, seed: int = 42):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        
        self.weights_input_hidden: List[List[float]] = []
        self.weights_hidden_output: List[List[float]] = []
        self.labels: List[Any] = []
        self.feature_means: List[float] = []
        self.feature_stds: List[float] = []

    def _sigmoid(self, x: float) -> float:
        """Função de ativação sigmoid com proteção contra overflow."""
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))
    
    def _sigmoid_derivative(self, y: float) -> float:
        """Derivada da sigmoid dado o output y = sigmoid(x)."""
        return y * (1.0 - y)
    
    def _normalize(self, features: List[float]) -> List[float]:
        """Normaliza features usando Z-score."""
        if not self.feature_means:
            return features
        return [
            (f - mean) / (std if std > 0 else 1.0)
            for f, mean, std in zip(features, self.feature_means, self.feature_stds)
        ]

    def _compute_normalization_params(self, train_set: Dataset) -> None:
        """Calcula média e desvio padrão para normalização."""
        all_features = [[float(x) for x in inst.features] for inst in train_set]
        n_samples = len(all_features)
        n_features = len(all_features[0])
        
        self.feature_means = [
            sum(sample[i] for sample in all_features) / n_samples
            for i in range(n_features)
        ]
        
        self.feature_stds = [
            math.sqrt(sum((sample[i] - self.feature_means[i]) ** 2 for sample in all_features) / n_samples)
            for i in range(n_features)
        ]

    def _initialize_weights(self, n_input: int, n_output: int) -> None:
        """Inicializa pesos usando Xavier initialization."""
        random.seed(self.seed)
        
        limit_ih = math.sqrt(6.0 / (n_input + self.n_hidden))
        self.weights_input_hidden = [
            [random.uniform(-limit_ih, limit_ih) for _ in range(self.n_hidden)]
            for _ in range(n_input + 1)
        ]
        
        limit_ho = math.sqrt(6.0 / (self.n_hidden + n_output))
        self.weights_hidden_output = [
            [random.uniform(-limit_ho, limit_ho) for _ in range(n_output)]
            for _ in range(self.n_hidden + 1)
        ]

    def fit(self, train_set: Dataset) -> None:
        self.labels = sorted(set(inst.label for inst in train_set))
        n_input = len(train_set[0].features)
        n_output = len(self.labels)
        
        self._compute_normalization_params(train_set)
        self._initialize_weights(n_input, n_output)
        
        for _ in range(self.epochs):
            for instance in train_set:
                normalized = self._normalize([float(x) for x in instance.features])
                inputs = normalized + [1.0]
                
                # Forward pass - camada oculta
                hidden_net = [
                    sum(inputs[i] * self.weights_input_hidden[i][j] for i in range(len(inputs)))
                    for j in range(self.n_hidden)
                ]
                hidden_out = [self._sigmoid(net) for net in hidden_net] + [1.0]
                
                # Forward pass - camada de saída
                output_net = [
                    sum(hidden_out[j] * self.weights_hidden_output[j][k] for j in range(len(hidden_out)))
                    for k in range(n_output)
                ]
                output = [self._sigmoid(net) for net in output_net]
                
                # Targets (one-hot encoding)
                targets = [1.0 if instance.label == self.labels[k] else 0.0 for k in range(n_output)]
                
                # Backpropagation - camada de saída
                output_errors = [targets[k] - output[k] for k in range(n_output)]
                output_deltas = [output_errors[k] * self._sigmoid_derivative(output[k]) for k in range(n_output)]
                
                # Backpropagation - camada oculta
                hidden_errors = [
                    sum(output_deltas[k] * self.weights_hidden_output[j][k] for k in range(n_output))
                    for j in range(self.n_hidden)
                ]
                hidden_deltas = [hidden_errors[j] * self._sigmoid_derivative(hidden_out[j]) for j in range(self.n_hidden)]
                
                # Atualização dos pesos
                for j in range(len(hidden_out)):
                    for k in range(n_output):
                        self.weights_hidden_output[j][k] += self.learning_rate * output_deltas[k] * hidden_out[j]
                
                for i in range(len(inputs)):
                    for j in range(self.n_hidden):
                        self.weights_input_hidden[i][j] += self.learning_rate * hidden_deltas[j] * inputs[i]

    def predict(self, features: List[Any]) -> Any:
        normalized = self._normalize([float(x) for x in features])
        inputs = normalized + [1.0]
        
        hidden_out = [
            self._sigmoid(sum(inputs[i] * self.weights_input_hidden[i][j] for i in range(len(inputs))))
            for j in range(self.n_hidden)
        ] + [1.0]
        
        output = [
            self._sigmoid(sum(hidden_out[j] * self.weights_hidden_output[j][k] for j in range(len(hidden_out))))
            for k in range(len(self.labels))
        ]
        
        return self.labels[output.index(max(output))]