import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Leitura do dataset
# -------------------------------------------------------------------
df = pd.read_csv(Path(__file__).resolve().parent.parent / "dataset" / "arsenio_dataset.csv")

print(df.head())  # Mostra as primeiras linhas
print()
print(df.info())  # Mostra informações do DataFrame
print()

# Seleção das variáveis
X = df[['Idade', 'Uso_Beber', 'Uso_Cozinhar', 'Arsenio_Agua']].values
y = df['Arsenio_Unhas'].values

# Adiciona o termo de intercepto (coluna de 1s)
X = np.c_[np.ones(X.shape[0]), X]
y = y.reshape(-1, 1)

# -------------------------------------------------------------------
# Questão (a) – Ajuste do modelo de regressão linear múltipla
# -------------------------------------------------------------------
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
XtY = X.T @ y
beta = XtX_inv @ XtY

print('Coeficientes (β):')
print(beta.flatten())
print()

# -------------------------------------------------------------------
# Questão (b) – Previsão para um caso específico
# -------------------------------------------------------------------
entrada = np.array([1, 30, 5, 5, 0.135])  # 1 para o intercepto
predicao = entrada @ beta
print('Previsão (arsênio na unha):', predicao.item())
print()

# -------------------------------------------------------------------
# Questão (c, d, e, f) – Avaliação do modelo
# -------------------------------------------------------------------
y_pred = X @ beta

# (d) R² e R² ajustado
SS_res = np.sum((y - y_pred)**2)
SS_tot = np.sum((y - np.mean(y))**2)
R2 = 1 - (SS_res/SS_tot)

n = X.shape[0]
p = X.shape[1] - 1
R2_adj = 1 - (1-R2)*(n-1)/(n-p-1)

print('R²:', R2)
print('R² ajustado:', R2_adj)
print()

# (f) Análise de resíduos
residuos = y - y_pred
tabela = pd.DataFrame({
    'Real': y.flatten(),
    'Previsto': y_pred.flatten(),
    'Residuo': residuos.flatten()
})
print(tabela.head())
print()

plt.scatter(y_pred, residuos)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valores Previstos')
plt.ylabel('Resíduos')
plt.title('Análise de Resíduos')
plt.show()

# -------------------------------------------------------------------
# Questão (a, b, c) – Cenário com intercepto forçado a 0
# -------------------------------------------------------------------
X_no_intercept = X[:,1:]  # remove a coluna de 1s
beta_no_intercept = np.linalg.inv(X_no_intercept.T @ X_no_intercept) @ (X_no_intercept.T @ y)
y_pred_no_intercept = X_no_intercept @ beta_no_intercept

SS_res2 = np.sum((y - y_pred_no_intercept)**2)
R2_no_intercept = 1 - SS_res2/SS_tot

print('R² sem intercepto:', R2_no_intercept)
print()

# -------------------------------------------------------------------
# Questão (g, h) – Métricas de erro
# -------------------------------------------------------------------
MSE = np.mean((y - y_pred) ** 2)
RMSE = np.sqrt(MSE)
MAE = np.mean(np.abs(y - y_pred))

print('Métricas de erro:')
print(f'  MSE: {MSE:.4f}')
print(f'  RMSE: {RMSE:.4f}')
print(f'  MAE: {MAE:.4f}')
