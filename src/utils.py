"""
Módulo de utilidades para o projeto de detecção de phishing.
Contém funções auxiliares utilizadas em diferentes partes do projeto.
"""

import re
import logging
from urllib.parse import urlparse


def setup_logging(level=logging.INFO):
    """
    Configura o sistema de logging.
    
    Args:
        level: Nível de logging (padrão: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_url(url):
    """
    Valida se a string fornecida é uma URL válida.
    
    Args:
        url: String a ser validada
        
    Returns:
        Boolean indicando se a URL é válida
    """
    # Verificar se a URL está vazia
    if not url:
        return False
    
    # Padrão básico para URLs
    pattern = re.compile(
        r'^(?:http|https)://'  # http:// ou https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domínio
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...ou endereço IP
        r'(?::\d+)?'  # porta opcional
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return bool(pattern.match(url))


def extract_domain(url):
    """
    Extrai o domínio de uma URL.
    
    Args:
        url: URL a ser processada
        
    Returns:
        String contendo o domínio
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        return domain
    except:
        return ""


def is_ip_address(domain):
    """
    Verifica se o domínio é um endereço IP.
    
    Args:
        domain: Domínio a ser verificado
        
    Returns:
        Boolean indicando se o domínio é um IP
    """
    ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
    return bool(re.match(ip_pattern, domain))


def count_suspicious_elements(url):
    """
    Conta elementos suspeitos em uma URL que podem indicar phishing.
    
    Args:
        url: URL a ser analisada
        
    Returns:
        Contagem de elementos suspeitos
    """
    suspicious_count = 0
    
    # Lista de elementos suspeitos
    suspicious_elements = ['@', 'login', 'signin', 'account', 'banking', 'secure', 'update', 
                          'service', 'confirm', 'password', 'verify']
    
    # Verificar cada elemento
    for element in suspicious_elements:
        if element in url.lower():
            suspicious_count += 1
    
    return suspicious_count
