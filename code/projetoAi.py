# %%
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "dataset" / "arsenio_dataset.csv")
df

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
nomes = ['Intercepto', 'Idade', 'Uso_Beber', 'Uso_Cozinhar', 'Arsenio_Agua']
for nome, valor in zip(nomes, beta.flatten()):
    print(f'{nome}: {valor}')

"""
------------------------------------------------------------------------------------
Questão (b) Use o modelo para prever o arsênio nas unhas, quando a idade for 30, a
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

"""
------------------------------------------------------------------------------------
Questão (g) Cenário com Intercepto Forçado a Zero: Suponha que, por razões teóricas,
você imponha a restrição de que o intercepto deve ser zero. Qual é a interpretação 
prática dessa decisão? Compare as métricas R² e RMSE deste modelo com o modelo com 
intercepto. Qual você escolheria e por quê?
------------------------------------------------------------------------------------
  > Forçar o intercepto a ser zero implica que quando todas as variáveis independen-
  tes são zero, a variável dependente também deve ser zero. Na prática, isso pode 
  não fazer sentido, especialmente se o ponto onde todas as variáveis independentes 
  são zero não for realista ou relevante para o contexto do problema. Por exemplo, 
  no caso do arsênio nas unhas, ter idade zero e não usar água para beber ou cozi-
  nhar pode não ser uma situação plausível.
  
  > Comparando as métricas R² e RMSE do modelo com intercepto e do modelo sem inter-
  cepto, geralmente o modelo com intercepto tende a ter um ajuste melhor (maior R²
  e menor RMSE), pois ele tem mais flexibilidade para se ajustar aos dados.
------------------------------------------------------------------------------------
"""
X_no_intercept = X[:,1:]  # remove a coluna de 1s
beta_no_intercept = np.linalg.inv(X_no_intercept.T @ X_no_intercept) @ (X_no_intercept.T @ Y)
y_pred_no_intercept = X_no_intercept @ beta_no_intercept

SS_res2 = np.sum((Y - y_pred_no_intercept)**2)
R2_no_intercept = 1 - SS_res2/SS_tot

MSE_normal = np.mean((Y - y_pred) ** 2)
RMSE_normal = np.sqrt(MSE_normal)

MSE_no_intercept = np.mean((Y - y_pred_no_intercept) ** 2)
RMSE_no_intercept = np.sqrt(MSE_no_intercept)

print(organizacao_console)
print('Comparação com o modelo com intercepto:')
print(f' - R² com intercepto: {R2}, R² sem intercepto: {R2_no_intercept}')
print(f' - RMSE com intercepto: {RMSE_normal}, RMSE sem intercepto: {RMSE_no_intercept}')

"""
------------------------------------------------------------------------------------
Questão (h) Além do R²: Calcule e interprete pelo menos outras duas métricas de erro
para o modelo completo e para o modelo alternativo (apenas com Arsênio na água). 

Sugestões: Erro Quadrático Médio (MSE), Raiz do Erro Quadrático Médio (RMSE) e 
Erro Absoluto Médio (MAE).
------------------------------------------------------------------------------------
  > As métricas adicionais fornecem uma visão mais completa do desempenho do modelo.
  O MSE penaliza mais os erros maiores, o RMSE traz a penalização para a mesma uni-
  dade da variável resposta, e o MAE dá uma medida direta do erro médio absoluto.
  
  > Comparando essas métricas entre o modelo completo e o modelo alternativo, podemos
  avaliar qual modelo tem um desempenho melhor em termos de precisão das previsões.
------------------------------------------------------------------------------------
"""
MAE_normal = np.mean(np.abs(Y - y_pred))

MSE_alt = np.mean((Y - y_pred_alt) ** 2)
RMSE_alt = np.sqrt(MSE_alt)
MAE_alt = np.mean(np.abs(Y - y_pred_alt))

print(organizacao_console)
print('[Comparação de métricas adicionais]')
print(f'Modelo completo: \n- MSE: {MSE_normal} \n- RMSE: {RMSE_normal} \n- MAE: {MAE_normal}')
print(f'Modelo alternativo: \n- MSE: {MSE_alt} \n- RMSE: {RMSE_alt} \n- MAE: {MAE_alt}')
print(organizacao_console)
# %%
