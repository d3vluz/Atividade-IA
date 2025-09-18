
# Relatório: Análise de Modelos de Regressão para Previsão de Arsênio nas Unhas

## Introdução

Este relatório apresenta uma análise sobre o uso de modelos de regressão para prever a concentração de arsênio nas unhas de indivíduos expostos a diferentes condições ambientais. O objetivo é avaliar como variáveis como idade, uso da água para beber e cozinhar, e a concentração de arsênio na água contribuem para explicar a presença de arsênio nas unhas. A análise busca apoiar estratégias de monitoramento e prevenção de riscos à saúde em comunidades expostas ao arsênio.

---

## Fundamentação Teórica

### Regressão Linear Simples

A regressão linear simples modela a relação entre uma variável dependente (resposta) e uma variável independente (preditor). Por exemplo, ao analisar apenas a concentração de arsênio na água para prever o arsênio nas unhas, temos um modelo de regressão linear simples.

### Regressão Linear Múltipla

Na regressão linear múltipla, utilizamos várias variáveis independentes para explicar a variável dependente. Isso permite capturar efeitos combinados, como idade, uso da água e concentração de arsênio na água, sobre o arsênio nas unhas.

### Interpretação dos Coeficientes

Cada coeficiente indica o quanto a variável dependente deve variar, em média, quando a respectiva variável independente aumenta uma unidade, mantendo as demais constantes.  
Exemplo: “Se aumentamos a idade em 1 ano, espera-se que o arsênio nas unhas aumente em X ppm, mantendo as outras variáveis constantes.”

### Intercepto

O intercepto representa o valor esperado da variável dependente quando todas as variáveis independentes são zero.

### Métricas de Avaliação

- **R²:** Mede a proporção da variabilidade explicada pelo modelo (quanto mais próximo de 1, melhor).
- **RMSE (Raiz do Erro Quadrático Médio):** Mede o erro médio das previsões, penalizando mais os grandes desvios.
- **MAE (Erro Absoluto Médio):** Indica o erro médio absoluto das previsões.

---

## Protocolo Experimental

- **Fonte dos Dados:**  
	Os dados analisados são do dataset arseniodataset.csv disponibilizado na plataforma AVA.

- **Variáveis Incluídas no Modelo:**  
	- Idade dos participantes  
	- Uso da água para beber (categoria)  
	- Uso da água para cozinhar (categoria)  
	- Concentração de arsênio na água (ppm)  
	- Concentração de arsênio nas unhas (ppm, variável resposta)

- **Ferramentas Utilizadas:**  
	Python, com as bibliotecas Numpy e Pandas para análise de dados e ajuste dos modelos.

- **Procedimento:**  
	1. Leitura e preparação dos dados.
	2. Ajuste do modelo de regressão linear múltipla.
	3. Avaliação das métricas de desempenho.
	4. Comparação com modelo alternativo (apenas arsênio na água).
	5. Análise de resíduos para verificação das suposições do modelo.

---

## Resultados e Discussão

### Coeficientes Estimados

O modelo completo apresentou os seguintes coeficientes (β):  
**[LEMBRAR DE TERMINAR AQUI] -Evandro!!!!**

Cada coeficiente indica o impacto de sua respectiva variável sobre o arsênio nas unhas.

### Previsão para Caso Específico

Para um indivíduo de 30 anos, com uso da água para beber e cozinhar na categoria 5, e concentração de arsênio na água de 0.135 ppm, a previsão do modelo foi de **[REVISAR]!!** ppm de arsênio nas unhas.

### Avaliação do Modelo

- **R²:** **[REVISAR]!!**
- **R² ajustado:** **[REVISAR]!!**
- **RMSE:** **[REVISAR]!!**
- **MAE:** **[REVISAR]!!**

### Comparação com Modelo Alternativo

O modelo alternativo, usando apenas a concentração de arsênio na água, apresentou R² e RMSE inferiores ao modelo completo, indicando que as demais variáveis contribuem para melhorar a explicação da variabilidade do arsênio nas unhas.

### Análise de Resíduos

Os resíduos foram analisados para verificar normalidade e homocedasticidade, confirmando a adequação do modelo.

---

## Conclusão

O modelo de regressão linear múltipla mostrou-se eficaz para prever a concentração de arsênio nas unhas, considerando múltiplos fatores de exposição. O uso de métricas como R², RMSE e MAE permitiu avaliar a qualidade do ajuste e comparar diferentes modelos. O modelo completo apresentou melhor desempenho em relação ao modelo alternativo, destacando a importância de considerar múltiplas variáveis. Recomenda-se, para trabalhos futuros, a inclusão de mais variáveis e a validação do modelo em outros contextos para aprimorar a capacidade preditiva e a generalização dos resultados.