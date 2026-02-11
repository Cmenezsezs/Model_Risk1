# Análise de Risco de Modelo: Credit Default (Logistic Regression vs. XGBoost)

Este projeto implementa um pipeline de **Validação de Risco de Modelo (Model Risk Management - MRM)** aplicado à concessão de crédito. O objetivo não é apenas maximizar a acurácia, mas avaliar a robustez, discriminação e, crucialmente, a calibração das probabilidades de default (PD).

O script compara o padrão da indústria bancária clássica (**Regressão Logística**) com algoritmos de boosting baseados em árvores (**XGBoost**), utilizando dados sintéticos desbalanceados que simulam um portfólio de crédito real.

##  Visão Geral da Análise

A análise se baseia em três pilares fundamentais para modelos financeiros regulados:

1.  **Discriminação (Poder de Separação):** Capacidade do modelo de distinguir bons de maus pagadores (Métricas: ROC-AUC, Gini).
2.  **Calibração (Confiabilidade da Probabilidade):** A probabilidade predita (ex: 20%) reflete a taxa de evento real? Fundamental para cálculos de Perda Esperada (EL) e precificação (Pricing).
3.  **Estabilidade de Distribuição:** Avaliação da sobreposição de scores (Brier Score).

##  Interpretação dos Resultados (Model Risk Analysis)

Ao executar o script, o painel comparativo revela insights críticos sobre o *trade-off* entre complexidade e confiabilidade:

### 1. Discriminação (ROC-AUC e Gini)
*   **Expectativa:** O **XGBoost** geralmente apresenta um Gini superior devido à sua capacidade de capturar relações não-lineares complexas e interações entre variáveis que a Regressão Logística (linear no *logit*) perde.
*   **Risco de Modelo:** Um Gini excessivamente alto em dados de teste pode sinalizar *overfitting*. Em crédito, modelos "perfeitos" são suspeitos e muitas vezes falham em produção (OOT).

### 2. Calibração (Curva de Calibração / Reliability Diagram)
*   **Ponto Crítico:** Este é o diferencial para instituições financeiras.
*   **Regressão Logística:** Tende a ser **bem calibrada** nativamente (a curva segue a diagonal). Suas probabilidades são diretamente interpretáveis como taxas de risco.
*   **XGBoost:** Frequentemente apresenta uma curva em forma de "S" (subestima riscos baixos e superestima riscos altos).
*   **Impacto no Negócio:** Um modelo com alto Gini mas má calibração gera prejuízo. Se o modelo diz que o risco é 5% (mas o real é 10%), o banco cobrará juros insuficientes para cobrir a inadimplência.

### 3. Conclusão Técnica
Para motores de decisão de crédito onde a explicabilidade e a calibração precisa da PD são mandatórias (ex: Basileia, IFRS 9), a **Regressão Logística** permanece um benchmark robusto. O **XGBoost** requer etapas adicionais de calibração (ex: *Isotonic Regression* ou *Platt Scaling*) para ser seguro em produção.


##  Tecnologias Utilizadas

*   **Python 3.8+**
*   `scikit-learn`: Geração de dados, Regressão Logística e métricas.
*   `xgboost`: Implementação de Gradient Boosting.
*   `matplotlib`: Visualização do painel de risco.
*   `numpy` & `pandas`: Manipulação vetorial e tabular.

##  Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/SEU-USUARIO/credit-model-risk.git
   
2. Instale as dependências:

bash
pip install pandas numpy matplotlib scikit-learn xgboost

3. Execute o script de análise:

bash
python model_risk_analysis.py

O script gera um painel com 3 gráficos essenciais:

 1. Curva ROC: Comparativo de sensibilidade vs. especificidade.

 2. Diagrama de Confiabilidade: Avaliação visual da calibração (Curva ideal = Diagonal perfeita).

 3. Histograma de Densidade: Visualização da separação das classes (KS visual).
