import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

df = pd.read_csv(Path(__file__).resolve().parent.parent / "dataset" / "arsenio_dataset.csv")

# print(df.head())
# print(df.info())

X = df[['Idade', 'Uso_Beber', 'Uso_Cozinhar', 'Arsenio_Agua']].values
Y = df['Arsenio_Unhas'].values

# Adiciona o termo de intercepto (coluna de 1s)
X = np.c_[np.ones(X.shape[0]), X]
Y = Y.reshape(-1, 1)
organizacao_console = 70*'-'

"""
------------------------------------------------------------------------------------
Questão (a) Ajuste do modelo de regressão linear múltipla usando a concentração de 
arsênio nas unhas como a resposta, e idade, uso para beber, uso para cozinhar e con-
centração de arsênio na água como os regressores.
------------------------------------------------------------------------------------
    > Para resolução desse problema foi necessário utilizar a fórmula dos mínimos
      quadrados: β = (XᵀX)⁻¹XᵀY
------------------------------------------------------------------------------------
"""

XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
XtY = X.T @ Y
beta = XtX_inv @ XtY

print(organizacao_console)
print('Coeficientes (β):')
print(beta.flatten())

"""
------------------------------------------------------------------------------------
Questão (b) Use o modelo para prever o arênio nas unhas, quando a idade for 30, a
categoria da água para beber for 5, a categoria da água para cozinhar for 5 e o ar-
sénio na água for 0.135 ppm.
------------------------------------------------------------------------------------
    > Para fazer a previsão, basta multiplicar os valores dos regressores pelo vetor
      de coeficientes (β) obtido anteriormente.
------------------------------------------------------------------------------------
"""

entrada = np.array([1, 30, 5, 5, 0.135])  # 1 para o intercepto
predicao = entrada @ beta

print(organizacao_console)
print('Previsão (arsênio na unha):', predicao.item())

"""
------------------------------------------------------------------------------------
Questão (c) Cálculo do coeficiente de determinação (R²) do modelo
------------------------------------------------------------------------------------
    > O coeficiente de determinação R². Ele é calculado por:
        R² = 1 - (SS_res / SS_tot)
      onde:
        - SS_res: soma dos quadrados dos resíduos 
        - SS_tot: soma total dos quadrados
        
      Valores de R² próximos de 1 indicam bom ajuste do modelo.
------------------------------------------------------------------------------------
"""
y_pred = X @ beta

SS_res = np.sum((Y - y_pred)**2)
SS_tot = np.sum((Y - np.mean(Y))**2)
R2 = 1 - (SS_res/SS_tot)

print(organizacao_console)
print('R²:', R2)

"""
------------------------------------------------------------------------------------
Questão (d) Muitos usuários de regressão preferem usar a estatística do valor aju-
stado de R². Por quê? Ela foi melhor que o R² comum? Se sim, por quê?
------------------------------------------------------------------------------------
  > O R² ajustado é preferido porque leva em consideração o número de variáveis (re-
  gressores) do modelo e penaliza a inclusão de variáveis irrelevantes. Enquanto o R²
  comum sempre aumenta (ou permanece igual) com a adição de novos regressores, o R² 
  ajustado só aumenta se a nova variável realmente contribuir para explicar a varia-
  bilidade da resposta. Assim, o R² ajustado é uma métrica mais realista para compa-
  rar modelos com diferentes números de variáveis.
  
  > No modelo estudado, o valor do R² ajustado foi ligeiramente menor que o R² comum,
  o que é esperado, pois ele corrige o otimismo do R² simples. Se o R² ajustado for 
  significativamente menor que o R², pode indicar que há variáveis desnecessárias no
  modelo.
------------------------------------------------------------------------------------
"""
n = X.shape[0]
p = X.shape[1] - 1
R2_adj = 1 - (1-R2)*(n-1)/(n-p-1)

print(organizacao_console)
print('R² ajustado:', R2_adj)

"""
------------------------------------------------------------------------------------
Questão (e) Compare este modelo com um modelo alternativo que use apenas a concen-
traçao de arsênio na  ́agua como preditor. Qual modelo  ́e melhor? Por que?
------------------------------------------------------------------------------------
  > Para comparar os modelos, podemos ajustar um segundo modelo usando apenas a
  concentração de arsênio na água como preditor e calcular o R² e o R² ajustado.
------------------------------------------------------------------------------------
"""
X_alt = df[['Arsenio_Agua']].values
X_alt = np.c_[np.ones(X_alt.shape[0]), X_alt]

beta_alt = np.linalg.inv(X_alt.T @ X_alt) @ (X_alt.T @ Y)

y_pred_alt = X_alt @ beta_alt
SS_res_alt = np.sum((Y - y_pred_alt)**2)

R2_alt = 1 - SS_res_alt/SS_tot
R2_adj_alt = 1 - (1-R2_alt)*(n-1)/(n-1-1)

print(organizacao_console)
print('Modelo alternativo (apenas arsênio na água):')
print(' - Coeficientes (β):', beta_alt.flatten())
print(' - R²:', R2_alt)
print(' - R² ajustado:', R2_adj_alt)

"""
------------------------------------------------------------------------------------
Questão (f) Realize uma análise de resíduos para verificar as suposições do modelo 
de regressão. Para isso, siga os seguintes passos:

    I. Calcule os valores ajustados para todas as observações de y:
    II. Calcule os resíduos correspondentes:
    III. Organize os resultados em uma tabela mostrando, para cada observação.
------------------------------------------------------------------------------------
  > A análise de resíduos é fundamental para verificar as suposições do modelo de re-
  gressão linear, como a normalidade dos resíduos, homocedasticidade (variância con-
  stante) e independência dos erros. A seguir, realizamos essa análise.
------------------------------------------------------------------------------------
"""

# I. Calcule os valores ajustados para todas as observações de y:
y_pred = X @ beta

# II. Calcule os resíduos correspondentes:
residuos = Y - y_pred

# III. Organize os resultados em uma tabela mostrando, para cada observação.
tabela = pd.DataFrame({
    'Índice': np.arange(1, len(Y)+1),
    'Valor Observado': Y.flatten(),
    'Valor Ajustado': y_pred.flatten(),
    'Resíduo': residuos.flatten()
})

print(organizacao_console)
print(tabela.head())

# @TODO:revisar isso, não entendi isso, commit do @matheusperestrelo
# plt.scatter(y_pred, residuos)
# plt.axhline(0, color='red', linestyle='--')
# plt.xlabel('Valores Previstos')
# plt.ylabel('Resíduos')
# plt.title('Análise de Resíduos')
# plt.show()

# -------------------------------------------------------------------
# Questão (a, b, c) – Cenário com intercepto forçado a 0
# -------------------------------------------------------------------
X_no_intercept = X[:,1:]  # remove a coluna de 1s
beta_no_intercept = np.linalg.inv(X_no_intercept.T @ X_no_intercept) @ (X_no_intercept.T @ Y)
y_pred_no_intercept = X_no_intercept @ beta_no_intercept

SS_res2 = np.sum((Y - y_pred_no_intercept)**2)
R2_no_intercept = 1 - SS_res2/SS_tot

print('R² sem intercepto:', R2_no_intercept)
print()

# -------------------------------------------------------------------
# Questão (g, h) – Métricas de erro
# -------------------------------------------------------------------
MSE = np.mean((Y - y_pred) ** 2)
RMSE = np.sqrt(MSE)
MAE = np.mean(np.abs(Y - y_pred))

print('Métricas de erro:')
print(f'  MSE: {MSE:.4f}')
print(f'  RMSE: {RMSE:.4f}')
print(f'  MAE: {MAE:.4f}')
