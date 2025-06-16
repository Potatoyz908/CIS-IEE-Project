"""
Módulo de pré-processamento para o modelo de detecção de phishing.
Este módulo fornece funcionalidades para transformar URLs em features para o modelo.
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import CountVectorizer


def extract_url_features(url):
    """
    Extrai características básicas de uma URL.
    
    Args:
        url: URL a ser analisada
        
    Returns:
        Dicionário com as features extraídas
    """
    features = {}
    
    # Análise básica da URL
    parsed = urlparse(url)
    
    # Comprimento da URL
    features['url_length'] = len(url)
    
    # Domínio
    features['domain'] = parsed.netloc
    features['domain_length'] = len(parsed.netloc)
    
    # Caminho
    features['path'] = parsed.path
    features['path_length'] = len(parsed.path)
    
    # Número de subdomínios
    features['subdomain_count'] = len(parsed.netloc.split('.')) - 1
    
    # Presença de IP no lugar de domínio
    ip_pattern = r'\d+\.\d+\.\d+\.\d+'
    features['has_ip'] = 1 if re.search(ip_pattern, parsed.netloc) else 0
    
    # Presença de caracteres suspeitos
    features['has_at_symbol'] = 1 if '@' in url else 0
    features['has_double_slash_redirect'] = 1 if '//' in parsed.path else 0
    
    # Presença de https
    features['is_https'] = 1 if parsed.scheme == 'https' else 0
    
    return features


def get_webpage_features(url):
    """
    Extrai características do conteúdo da página web.
    
    Args:
        url: URL da página a ser analisada
        
    Returns:
        Dicionário com as features extraídas
    """
    features = {}
    
    try:
        # Definir um User-Agent para evitar bloqueios
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Obter o conteúdo da página
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Contagem de forms
        features['form_count'] = len(soup.find_all('form'))
        
        # Contagem de inputs
        features['input_count'] = len(soup.find_all('input'))
        
        # Presença de campos de senha
        features['has_password_field'] = 1 if soup.find('input', {'type': 'password'}) else 0
        
        # Extração de texto para análise
        features['page_text'] = ' '.join([p.get_text() for p in soup.find_all('p')])
        
        # Status da resposta
        features['response_status'] = response.status_code
        
    except Exception as e:
        # Em caso de erro, preencher com valores padrão
        features['form_count'] = 0
        features['input_count'] = 0
        features['has_password_field'] = 0
        features['page_text'] = ''
        features['response_status'] = 0
    
    return features


def preprocess_data(urls, extract_webpage_content=False):
    """
    Realiza o pré-processamento completo das URLs.
    
    Args:
        urls: Lista de URLs a serem processadas
        extract_webpage_content: Se True, também extrai conteúdo das páginas web
        
    Returns:
        DataFrame com todas as features processadas
    """
    features_list = []
    
    for url in urls:
        # Extrair características da URL
        url_features = extract_url_features(url)
        
        # Opcionalmente extrair características da página web
        if extract_webpage_content:
            webpage_features = get_webpage_features(url)
            # Combinar os dicionários
            all_features = {**url_features, **webpage_features}
        else:
            all_features = url_features
        
        # Adicionar URL original como referência
        all_features['url'] = url
        
        features_list.append(all_features)
    
    # Converter para DataFrame
    df = pd.DataFrame(features_list)
    
    # Processar texto (se disponível)
    if extract_webpage_content and 'page_text' in df.columns:
        vectorizer = CountVectorizer(max_features=100)
        text_features = vectorizer.fit_transform(df['page_text'].fillna(''))
        text_df = pd.DataFrame(text_features.toarray(), 
                               columns=[f'text_{i}' for i in range(text_features.shape[1])])
        
        # Concatenar features de texto
        df = pd.concat([df.drop('page_text', axis=1), text_df], axis=1)
    
    return df


def prepare_for_model(df, model_columns):
    """
    Prepara o DataFrame para ser usado no modelo, garantindo que tenha as colunas corretas.
    
    Args:
        df: DataFrame com as features extraídas
        model_columns: Lista de colunas esperadas pelo modelo
        
    Returns:
        DataFrame formatado para uso no modelo
    """
    # Salvar a URL para referência (não é uma feature para o modelo)
    urls = df['url'] if 'url' in df.columns else None
    
    # Remover colunas que não são numéricas ou que não devem ser usadas pelo modelo
    non_feature_cols = ['url', 'domain', 'path', 'page_text']
    for col in non_feature_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Garantir que todas as colunas necessárias existam
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Manter apenas as colunas esperadas pelo modelo, na ordem correta
    df = df[model_columns]
    
    return df, urls
