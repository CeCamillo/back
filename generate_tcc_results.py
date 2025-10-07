import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

# Definir seed aleatória para reprodutibilidade
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configurar matplotlib para saídas de alta qualidade
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 8)

# Criar diretório de saída
OUTPUT_DIR = Path("tcc_results")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("SCRIPT DE GERAÇÃO DE RESULTADOS TCC")
print("Análise Comparativa: Modelo Ingênuo vs. Modelo Melhorado")
print("="*80)

# ============================================================================
# PASSO 1: CARREGAR E PRÉ-PROCESSAR DADOS
# ============================================================================

print("\n[PASSO 1] Carregando e pré-processando dados...")

# Carregar datasets
print("  → Carregando conjunto de treinamento...")
df_train = pd.read_parquet("./datasets/UNSW_NB15_training-set.parquet")
print(f"     Conjunto de treinamento carregado: {len(df_train):,} amostras")

print("  → Carregando conjunto de teste...")
df_test = pd.read_parquet("./datasets/UNSW_NB15_testing-set.parquet")
print(f"     Conjunto de teste carregado: {len(df_test):,} amostras")


def preprocess_data(df):
    """
    Pré-processar o dataset UNSW-NB15.

    Passos:
    1. Filtrar para manter apenas classes Normal, Backdoor e Worms
    2. Criar coluna attack_label a partir de attack_cat
    3. Remover colunas de metadados (id, label, attack_cat)
    4. Retornar features (X) e rótulos (y)
    """
    # Filtrar dataset para focar em Normal, Backdoor e Worms
    df_filtered = df[df['attack_cat'].isin(['Worms', 'Backdoor']) | (df['label'] == 0)].copy()

    # Criar rótulo de ataque (Normal para label=0, caso contrário usar attack_cat)
    df_filtered['attack_label'] = df_filtered['attack_cat'].fillna('Normal')

    # Remover colunas de metadados
    df_filtered = df_filtered.drop(columns=[c for c in ['id', 'label', 'attack_cat'] if c in df_filtered])

    # Separar features e alvo
    X = df_filtered.drop(columns=['attack_label'])
    y = df_filtered['attack_label']

    return X, y


print("  → Pré-processando dados de treinamento...")
X_train_raw, y_train = preprocess_data(df_train)
print(f"     Formato das features de treinamento: {X_train_raw.shape}")
print(f"     Distribuição de classes (treinamento):")
for class_name, count in y_train.value_counts().items():
    print(f"       - {class_name}: {count:,} ({count/len(y_train)*100:.2f}%)")

print("  → Pré-processando dados de teste...")
X_test_raw, y_test = preprocess_data(df_test)
print(f"     Formato das features de teste: {X_test_raw.shape}")
print(f"     Distribuição de classes (teste):")
for class_name, count in y_test.value_counts().items():
    print(f"       - {class_name}: {count:,} ({count/len(y_test)*100:.2f}%)")

# Codificar features categóricas com one-hot encoding
print("  → Aplicando one-hot encoding às features categóricas...")
X_train = pd.get_dummies(X_train_raw)
X_test = pd.get_dummies(X_test_raw)

# Alinhar colunas de treino e teste (garantir que tenham as mesmas features)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
print(f"     Contagem final de features após codificação: {X_train.shape[1]}")

# Codificar rótulos
print("  → Codificando rótulos alvo...")
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
class_labels = le.classes_
print(f"     Classes codificadas: {list(class_labels)}")

print("\n✓ Pré-processamento de dados completo!\n")

# ============================================================================
# PASSO 2: TREINAR MODELO INGÊNUO (sem tratamento de desbalanceamento de classes)
# ============================================================================

print("[PASSO 2] Treinando Modelo Ingênuo (sem tratamento de desbalanceamento de classes)...")

# Criar pipeline: StandardScaler → RandomForest (sem balanceamento de classes)
naive_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        # SEM parâmetro class_weight - esta é a abordagem ingênua
        n_jobs=-1
    ))
])

print("  → Ajustando Modelo Ingênuo aos dados de treinamento...")
naive_pipeline.fit(X_train, y_train_encoded)

print("  → Fazendo predições no conjunto de teste...")
y_pred_naive = naive_pipeline.predict(X_test)

print("✓ Treinamento do Modelo Ingênuo completo!\n")

# ============================================================================
# PASSO 3: TREINAR MODELO MELHORADO (com class_weight='balanced')
# ============================================================================

print("[PASSO 3] Treinando Modelo Melhorado (com class_weight='balanced')...")

# Criar pipeline: StandardScaler → RandomForest (com balanceamento de classes)
improved_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight='balanced',  # ESTA É A MELHORIA CHAVE
        n_jobs=-1
    ))
])

print("  → Ajustando Modelo Melhorado aos dados de treinamento...")
improved_pipeline.fit(X_train, y_train_encoded)

print("  → Fazendo predições no conjunto de teste...")
y_pred_improved = improved_pipeline.predict(X_test)

print("✓ Treinamento do Modelo Melhorado completo!\n")

# ============================================================================
# PASSO 4: GERAR RELATÓRIOS DE CLASSIFICAÇÃO
# ============================================================================

print("[PASSO 4] Gerando relatórios de classificação...")

# Gerar relatórios de classificação completos
naive_report = classification_report(
    y_test_encoded,
    y_pred_naive,
    target_names=class_labels,
    digits=4
)

improved_report = classification_report(
    y_test_encoded,
    y_pred_improved,
    target_names=class_labels,
    digits=4
)

# Salvar relatórios em arquivos de texto
naive_report_path = OUTPUT_DIR / "naive_model_report.txt"
with open(naive_report_path, 'w') as f:
    f.write("MODELO INGÊNUO - RELATÓRIO DE CLASSIFICAÇÃO\n")
    f.write("="*60 + "\n")
    f.write("Modelo: RandomForestClassifier (n_estimators=100)\n")
    f.write("Tratamento de Desbalanceamento de Classes: Nenhum\n")
    f.write("="*60 + "\n\n")
    f.write(naive_report)
print(f"  ✓ Salvo: {naive_report_path}")

improved_report_path = OUTPUT_DIR / "improved_model_report.txt"
with open(improved_report_path, 'w') as f:
    f.write("MODELO MELHORADO - RELATÓRIO DE CLASSIFICAÇÃO\n")
    f.write("="*60 + "\n")
    f.write("Modelo: RandomForestClassifier (n_estimators=100)\n")
    f.write("Tratamento de Desbalanceamento de Classes: class_weight='balanced'\n")
    f.write("="*60 + "\n\n")
    f.write(improved_report)
print(f"  ✓ Salvo: {improved_report_path}")

print("\n✓ Relatórios de classificação salvos!\n")

# ============================================================================
# PASSO 5: GERAR MATRIZES DE CONFUSÃO
# ============================================================================

print("[PASSO 5] Gerando visualizações de matriz de confusão...")

# Gerar matrizes de confusão
cm_naive = confusion_matrix(y_test_encoded, y_pred_naive)
cm_improved = confusion_matrix(y_test_encoded, y_pred_improved)

# Normalizar matrizes de confusão para melhor visualização (porcentagens)
cm_naive_norm = cm_naive.astype('float') / cm_naive.sum(axis=1)[:, np.newaxis]
cm_improved_norm = cm_improved.astype('float') / cm_improved.sum(axis=1)[:, np.newaxis]

# Plotar Matriz de Confusão do Modelo Ingênuo
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm_naive_norm,
    annot=True,
    fmt='.2%',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels,
    cbar_kws={'label': 'Porcentagem'},
    ax=ax
)
ax.set_title('Matriz de Confusão - Modelo Ingênuo',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Classe Prevista', fontsize=12, fontweight='bold')
ax.set_ylabel('Classe Real', fontsize=12, fontweight='bold')
plt.tight_layout()

naive_cm_path = OUTPUT_DIR / "naive_model_confusion_matrix.png"
plt.savefig(naive_cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {naive_cm_path}")

# Plotar Matriz de Confusão do Modelo Melhorado
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm_improved_norm,
    annot=True,
    fmt='.2%',
    cmap='Greens',
    xticklabels=class_labels,
    yticklabels=class_labels,
    cbar_kws={'label': 'Porcentagem'},
    ax=ax
)
ax.set_title('Matriz de Confusão - Modelo Melhorado',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Classe Prevista', fontsize=12, fontweight='bold')
ax.set_ylabel('Classe Real', fontsize=12, fontweight='bold')
plt.tight_layout()

improved_cm_path = OUTPUT_DIR / "improved_model_confusion_matrix.png"
plt.savefig(improved_cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {improved_cm_path}")

print("\n✓ Matrizes de confusão salvas!\n")

# ============================================================================
# PASSO 6: GERAR ANÁLISE COMPARATIVA DE F1-SCORE
# ============================================================================

print("[PASSO 6] Gerando análise comparativa de F1-Score...")

# Obter relatórios de classificação detalhados como dicionários
naive_report_dict = classification_report(
    y_test_encoded,
    y_pred_naive,
    target_names=class_labels,
    output_dict=True
)

improved_report_dict = classification_report(
    y_test_encoded,
    y_pred_improved,
    target_names=class_labels,
    output_dict=True
)

# Extrair F1-scores para Backdoor e Worms
print("\n" + "="*60)
print("ANÁLISE COMPARATIVA DE F1-SCORE")
print("="*60)

for attack_type in ['Backdoor', 'Worms']:
    naive_f1 = naive_report_dict[attack_type]['f1-score']
    improved_f1 = improved_report_dict[attack_type]['f1-score']
    improvement = ((improved_f1 - naive_f1) / naive_f1 * 100) if naive_f1 > 0 else 0

    print(f"\nTipo de Ataque: {attack_type}")
    print(f"  - F1-Score Modelo Ingênuo:    {naive_f1:.4f}")
    print(f"  - F1-Score Modelo Melhorado:  {improved_f1:.4f}")
    print(f"  - Melhoria:                   {improvement:+.2f}%")

print("\n" + "="*60)

# Também imprimir métricas gerais
print("\nAcurácia Geral:")
print(f"  - Modelo Ingênuo:    {naive_report_dict['accuracy']:.4f}")
print(f"  - Modelo Melhorado:  {improved_report_dict['accuracy']:.4f}")

print("\nF1-Score Médio Ponderado:")
print(f"  - Modelo Ingênuo:    {naive_report_dict['weighted avg']['f1-score']:.4f}")
print(f"  - Modelo Melhorado:  {improved_report_dict['weighted avg']['f1-score']:.4f}")

print("\n✓ Análise comparativa completa!\n")

# ============================================================================
# PASSO 7: GERAR VISUALIZAÇÃO DE IMPORTÂNCIA DE FEATURES
# ============================================================================

print("[PASSO 7] Gerando visualização de importância de features...")

# Extrair importâncias de features do Modelo Melhorado
classifier = improved_pipeline.named_steps['classifier']
feature_importances = classifier.feature_importances_

# Criar DataFrame com nomes de features e importâncias
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

# Obter top 15 features
top_15_features = feature_importance_df.head(15)

# Criar gráfico de barras horizontal
fig, ax = plt.subplots(figsize=(12, 8))

# Plotar barras
bars = ax.barh(
    range(len(top_15_features)),
    top_15_features['importance'].values,
    color='steelblue',
    edgecolor='navy',
    linewidth=1.5
)

# Personalizar gráfico
ax.set_yticks(range(len(top_15_features)))
ax.set_yticklabels(top_15_features['feature'].values, fontsize=10)
ax.set_xlabel('Importância', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Features Mais Importantes',
             fontsize=14, fontweight='bold', pad=20)
ax.invert_yaxis()  # Maior importância no topo

# Adicionar rótulos de valor nas barras
for i, (importance, bar) in enumerate(zip(top_15_features['importance'].values, bars)):
    ax.text(importance, bar.get_y() + bar.get_height()/2,
            f' {importance:.4f}',
            va='center', ha='left', fontsize=9)

# Adicionar grade para melhor legibilidade
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()

feature_importance_path = OUTPUT_DIR / "feature_importances.png"
plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {feature_importance_path}")

# Também salvar importâncias de features como CSV para referência
feature_importance_csv_path = OUTPUT_DIR / "feature_importances.csv"
feature_importance_df.to_csv(feature_importance_csv_path, index=False)
print(f"  ✓ Salvo: {feature_importance_csv_path}")

print("\n✓ Visualização de importância de features salva!\n")

# ============================================================================
# RESUMO FINAL
# ============================================================================

print("="*80)
print("TODOS OS RESULTADOS GERADOS COM SUCESSO!")
print("="*80)
print(f"\nTodas as saídas salvas em: {OUTPUT_DIR.absolute()}/\n")
print("Arquivos gerados:")
print("  1. naive_model_report.txt              - Relatório de classificação para Modelo Ingênuo")
print("  2. improved_model_report.txt           - Relatório de classificação para Modelo Melhorado")
print("  3. naive_model_confusion_matrix.png    - Visualização da matriz de confusão (Ingênuo)")
print("  4. improved_model_confusion_matrix.png - Visualização da matriz de confusão (Melhorado)")
print("  5. feature_importances.png             - Gráfico das top 15 importâncias de features")
print("  6. feature_importances.csv             - Dados completos de importância de features")
print("\n" + "="*80)