"""
Módulo de treinamento para o modelo de detecção de phishing.
Este módulo fornece funcionalidades para treinar o modelo usando AutoML.
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pycaret.classification import setup, compare_models, tune_model, save_model, finalize_model

# Caminho para os arquivos de dados e modelo
DATA_DIR = Path(__file__).parent.parent / "data"
DATASET_PATH = DATA_DIR / "phishing_dataset_CIS.csv"
MODEL_OUTPUT_PATH = DATA_DIR / "modelo_phishing_final.pkl"
COLUNAS_TREINAMENTO_PATH = DATA_DIR / "colunas_treinamento.pkl"
COLUNAS_AUXILIARES_PATH = DATA_DIR / "colunas_auxiliares.pkl"


def load_data():
    """
    Carrega os dados de treinamento.
    
    Returns:
        DataFrame com os dados carregados
    """
    # Verificar se o arquivo de dados existe
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {DATASET_PATH}")
    
    # Carregar os dados
    return pd.read_csv(DATASET_PATH)


def preprocess_training_data(df):
    """
    Pré-processa os dados para treinamento.
    
    Args:
        df: DataFrame com os dados brutos
        
    Returns:
        X: Features pré-processadas
        y: Target (alvo)
    """
    # Identificar a coluna target
    target_column = 'phishing'  # Ajuste conforme o nome da sua coluna target
    
    # Verificar se a coluna target existe
    if target_column not in df.columns:
        raise ValueError(f"Coluna target '{target_column}' não encontrada nos dados")
    
    # Separar features e target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Guardar lista de colunas para uso na inferência
    with open(COLUNAS_TREINAMENTO_PATH, 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    return X, y


def train_automl_model(X, y):
    """
    Treina um modelo usando AutoML do PyCaret.
    
    Args:
        X: Features
        y: Target (alvo)
        
    Returns:
        Modelo treinado
    """
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configurar o ambiente PyCaret
    print("Configurando ambiente de treinamento...")
    clf = setup(data=pd.concat([X_train, y_train], axis=1), target=y.name, 
                session_id=123, verbose=True, normalize=True, 
                remove_multicollinearity=True, multicollinearity_threshold=0.9)
    
    # Comparar modelos e selecionar o melhor
    print("Comparando modelos...")
    best_model = compare_models(verbose=True)
    
    # Otimizar hiperparâmetros
    print("Otimizando hiperparâmetros...")
    tuned_model = tune_model(best_model, optimize='f1')
    
    # Avaliar no conjunto de teste
    print("Avaliando modelo...")
    predictions = predict_model(tuned_model, data=X_test)
    
    # Exibir métricas
    print("Métricas de avaliação:")
    y_pred = predictions['prediction_label']
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precisão: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred):.4f}")
    
    # Finalizar o modelo (treinar com todos os dados)
    print("Finalizando modelo...")
    final_model = finalize_model(tuned_model)
    
    # Salvar o modelo
    print("Salvando modelo...")
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(final_model, f)
    
    print(f"Modelo salvo em: {MODEL_OUTPUT_PATH}")
    return final_model


def main():
    """Função principal para execução do treinamento."""
    print("Iniciando treinamento do modelo de detecção de phishing...")
    
    # Carregar dados
    df = load_data()
    print(f"Dados carregados: {df.shape[0]} amostras, {df.shape[1]} colunas")
    
    # Pré-processar dados
    X, y = preprocess_training_data(df)
    print(f"Dados pré-processados: {X.shape[1]} features")
    
    # Treinar modelo
    model = train_automl_model(X, y)
    print("Treinamento concluído com sucesso!")


if __name__ == "__main__":
    main()
