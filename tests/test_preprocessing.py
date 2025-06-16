"""
Testes para o módulo de pré-processamento.
"""

import unittest
import pandas as pd
from src.preprocessing import extract_url_features, prepare_for_model


class TestPreprocessing(unittest.TestCase):
    """Testes unitários para as funções de pré-processamento."""
    
    def test_extract_url_features(self):
        """Testa a extração de características de URLs."""
        # URL de teste
        test_url = "https://example.com/login.php"
        
        # Extrair features
        features = extract_url_features(test_url)
        
        # Verificar se as features básicas foram extraídas
        self.assertEqual(features['url_length'], len(test_url))
        self.assertEqual(features['domain'], "example.com")
        self.assertEqual(features['path'], "/login.php")
        self.assertEqual(features['is_https'], 1)
        
        # Testar URL com subdomínio
        test_url_subdomain = "https://mail.example.com/inbox"
        features_subdomain = extract_url_features(test_url_subdomain)
        self.assertEqual(features_subdomain['subdomain_count'], 1)
        
        # Testar URL com IP
        test_url_ip = "http://192.168.1.1/admin"
        features_ip = extract_url_features(test_url_ip)
        self.assertEqual(features_ip['has_ip'], 1)
        
        # Testar URL com símbolo @
        test_url_at = "http://example.com/user@domain"
        features_at = extract_url_features(test_url_at)
        self.assertEqual(features_at['has_at_symbol'], 1)
    
    def test_prepare_for_model(self):
        """Testa a preparação de dados para o modelo."""
        # Criar DataFrame de teste
        test_df = pd.DataFrame({
            'url': ['https://example.com', 'http://test.com'],
            'url_length': [18, 15],
            'domain': ['example.com', 'test.com'],
            'domain_length': [11, 8],
            'path': ['', ''],
            'path_length': [0, 0],
            'subdomain_count': [0, 0],
            'has_ip': [0, 0],
            'is_https': [1, 0]
        })
        
        # Definir colunas esperadas pelo modelo
        model_columns = ['url_length', 'domain_length', 'path_length', 
                         'subdomain_count', 'has_ip', 'is_https', 'extra_feature']
        
        # Preparar dados para o modelo
        prepared_df, urls = prepare_for_model(test_df, model_columns)
        
        # Verificar se as dimensões estão corretas
        self.assertEqual(prepared_df.shape[1], len(model_columns))
        self.assertEqual(prepared_df.shape[0], 2)
        
        # Verificar se as colunas foram criadas corretamente
        self.assertTrue('extra_feature' in prepared_df.columns)
        self.assertEqual(prepared_df['extra_feature'].sum(), 0)  # Deve estar preenchida com zeros
        
        # Verificar se as URLs originais foram preservadas
        self.assertEqual(list(urls), ['https://example.com', 'http://test.com'])


if __name__ == '__main__':
    unittest.main()
