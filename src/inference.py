"""
Módulo de inferência para o modelo de detecção de phishing.
Este módulo fornece funcionalidades para carregar o modelo treinado e realizar predições.

Exemplo de uso:
    from src.inference import predict
    
    # URL a ser analisada
    url = "http://exemplo.com"
    
    # Realizar previsão
    result = predict(url)
    print(f"A URL {url} {'é' if result else 'não é'} phishing.")
"""

import os
import pickle
import pandas as pd
from pathlib import Path

# Caminho para os arquivos de modelo e colunas
DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = DATA_DIR / "modelo_phishing_final.pkl"
COLUNAS_TREINAMENTO_PATH = DATA_DIR / "colunas_treinamento.pkl"
COLUNAS_AUXILIARES_PATH = DATA_DIR / "colunas_auxiliares.pkl"


def load_model():
    """Carrega o modelo treinado e as colunas necessárias para a inferência."""
    
    # Verificar se o arquivo do modelo existe
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {MODEL_PATH}")
    
    # Carregar o modelo
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    # Carregar as colunas de treinamento
    with open(COLUNAS_TREINAMENTO_PATH, "rb") as f:
        colunas_treinamento = pickle.load(f)
    
    # Carregar as colunas auxiliares (se existirem)
    colunas_auxiliares = None
    if COLUNAS_AUXILIARES_PATH.exists():
        with open(COLUNAS_AUXILIARES_PATH, "rb") as f:
            colunas_auxiliares = pickle.load(f)
    
    return model, colunas_treinamento, colunas_auxiliares


def preprocess_url(url):
    """
    Pré-processa a URL para gerar as features necessárias para o modelo.
    
    Esta é uma versão simplificada. Na implementação real, você usaria o mesmo
    pré-processamento usado durante o treinamento.
    
    Args:
        url: URL a ser analisada
        
    Returns:
        DataFrame com as features processadas
    """
    # Implementação simplificada
    # Em um cenário real, você importaria funções do módulo preprocessing.py
    
    # Criando um DataFrame vazio com as colunas esperadas pelo modelo
    _, colunas_treinamento, _ = load_model()
    
    # Criar um DataFrame simulado (substituir por pré-processamento real)
    features = pd.DataFrame([[0] * len(colunas_treinamento)], columns=colunas_treinamento)
    
    return features


def predict(url):
    """
    Realiza a predição para determinar se uma URL é phishing.
    
    Args:
        url: URL a ser analisada
        
    Returns:
        True se a URL for classificada como phishing, False caso contrário
    """
    # Carregar o modelo e as colunas
    model, _, _ = load_model()
    
    # Pré-processar a URL
    features = preprocess_url(url)
    
    # Realizar a predição
    prediction = model.predict(features)
    
    # Retornar o resultado (assumindo que 1 é phishing e 0 não é)
    return bool(prediction[0])


def main():
    """Função principal para demonstração ou execução via linha de comando."""
    print("Serviço de detecção de phishing iniciado!")
    
    # Aqui você poderia iniciar um serviço web, por exemplo
    print("Para usar este módulo, importe-o em seu código:")
    print("from src.inference import predict")
    print("result = predict('http://exemplo.com')")


if __name__ == "__main__":
    main()
