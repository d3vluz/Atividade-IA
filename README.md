
## Projeto: Análise de Regressão Linear - Arsênio em Unhas

Este projeto realiza uma análise de regressão linear múltipla para investigar fatores associados à concentração de arsênio nas unhas, utilizando dados fornecidos. O trabalho foi desenvolvido para a disciplina de Inteligência Artificial Computacional (T296-60).

### Estrutura do Projeto

- `code/projetoAi.py`: Script principal com toda a análise estatística e geração de gráficos.
- `dataset/arsenio_dataset.csv`: Base de dados utilizada na análise.

### Dependências

Para rodar o projeto, é necessário ter instalado:

- Python 3.7+
- numpy
- pandas
- matplotlib

Você pode instalar as dependências com o comando:

```bash
pip install numpy pandas matplotlib
```

### Como Executar

1. Certifique-se de que o arquivo `arsenio_dataset.csv` está na pasta `dataset/`.
2. Execute o script principal:

```bash
python code/projetoAi.py
```

O script irá:
- Ler o dataset
- Ajustar modelos de regressão linear múltipla e simples
- Exibir coeficientes, métricas de ajuste (R², R² ajustado, MSE, RMSE, MAE)
- Realizar análise de resíduos
- Comparar modelos com e sem intercepto

### Equipe

| Nome             | Matrícula |
|------------------|-----------|
| Evandro Luz      | 2220294   |
| Matheus Correia  | 2224701   |