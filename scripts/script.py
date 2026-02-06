import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve

# ---------------------------------------------------------
# 1. Geração de Dados Sintéticos de Crédito (Desbalanceados)
# ---------------------------------------------------------
np.random.seed(42)
print("Gerando dados de simulação de crédito...")

X, y = make_classification(
    n_samples=20000, 
    n_features=20, 
    n_informative=10, 
    n_redundant=5, 
    weights=[0.90, 0.10], # 90% Bons Pagadores, 10% Default (Realista)
    random_state=42
)

# Split de validação (OOT ou Hold-out é crítico para risco de modelo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# ---------------------------------------------------------
# 2. Treinamento dos Modelos
# ---------------------------------------------------------

# Modelo A: Regressão Logística (Benchmark de Estabilidade/Interpretabilidade)
lr_model = LogisticRegression(solver='liblinear', class_weight='balanced')
lr_model.fit(X_train, y_train)

# Modelo B: XGBoost (Benchmark de Performance/Não-Linearidade)
xgb_model = XGBClassifier(
    n_estimators=100, 
    max_depth=4, 
    learning_rate=0.1, 
    scale_pos_weight=9, # Compensar desbalanceamento
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train)

# ---------------------------------------------------------
# 3. Pipeline de Avaliação de Risco (Risk Engine)
# ---------------------------------------------------------

def analyze_model_risk(models_dict, X_test, y_test):
    plt.figure(figsize=(18, 6))
    
    # Plot 1: Curva ROC (Poder de Discriminação)
    ax1 = plt.subplot(1, 3, 1)
    
    # Plot 2: Curva de Calibração (Confiabilidade da PD)
    # Fundamental para Bancos: Se a curva não for diagonal, o banco perde dinheiro.
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot([0, 1], [0, 1], "k:", label="Perfeitamente Calibrado")
    
    # Plot 3: Densidade de Probabilidade (Separação das Classes)
    ax3 = plt.subplot(1, 3, 3)
    
    metrics_summary = []

    for name, model in models_dict.items():
        # Probabilidade de Default (PD)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 1. Métricas de Discriminação
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        gini = 2 * roc_auc - 1
        
        # 2. Métricas de Calibração (Brier Score)
        brier = brier_score_loss(y_test, y_prob)
        
        metrics_summary.append({
            "Modelo": name,
            "AUC": roc_auc,
            "Gini": gini,
            "Brier Score (Menor é melhor)": brier
        })
        
        # Plot ROC
        ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Plot Calibração
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)
        ax2.plot(mean_predicted_value, fraction_of_positives, "s-", label=f'{name}')
        
        # Plot Densidade
        ax3.hist(y_prob[y_test == 0], bins=50, alpha=0.3, label=f'{name} (Bons)', density=True)
        ax3.hist(y_prob[y_test == 1], bins=50, alpha=0.3, label=f'{name} (Default)', density=True, histtype='step', linewidth=2)

    # Estilização
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_title('Discriminação (Curva ROC)')
    ax1.set_xlabel('Taxa de Falsos Positivos')
    ax1.set_ylabel('Taxa de Verdadeiros Positivos')
    ax1.legend()
    
    ax2.set_title('Calibração (Reliability Diagram)')
    ax2.set_xlabel('Probabilidade Média Predita')
    ax2.set_ylabel('Fração de Positivos (Real)')
    ax2.legend()
    
    ax3.set_title('Distribuição de Scores (Sobreposição)')
    ax3.set_xlabel('Probabilidade de Default')
    ax3.set_ylabel('Densidade')
    
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame(metrics_summary)

# Executando a Análise
results = analyze_model_risk(
    {"Logistic Regression": lr_model, "XGBoost": xgb_model}, 
    X_test, y_test
)

print("\n--- Resumo de Métricas de Risco ---")
print(results)

