"""
Testes para o módulo de inferência.
"""

import unittest
import os
import pickle
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestInference(unittest.TestCase):
    """Testes unitários para as funções de inferência."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Mock do modelo
        self.mock_model = MagicMock()
        
        # Mock das colunas de treinamento
        self.mock_columns = ['url_length', 'domain_length', 'has_ip', 'is_https']
    
    @patch('src.inference.pickle.load')
    def test_load_model(self, mock_pickle_load):
        """Testa o carregamento do modelo."""
        from src.inference import load_model
        
        # Configurar os mocks para retornar valores específicos
        mock_pickle_load.side_effect = [self.mock_model, self.mock_columns, None]
        
        # Chamar a função
        model, columns, aux_columns = load_model()
        
        # Verificar se pickle.load foi chamado 3 vezes
        self.assertEqual(mock_pickle_load.call_count, 3)
        
        # Verificar os valores retornados
        self.assertEqual(model, self.mock_model)
        self.assertEqual(columns, self.mock_columns)
        self.assertIsNone(aux_columns)
    
    @patch('src.inference.load_model')
    def test_preprocess_url(self, mock_load_model):
        """Testa o pré-processamento de URLs."""
        from src.inference import preprocess_url
        
        # Configurar o mock para retornar valores específicos
        mock_load_model.return_value = (self.mock_model, self.mock_columns, None)
        
        # URL de teste
        test_url = "https://example.com"
        
        # Chamar a função
        features = preprocess_url(test_url)
        
        # Verificar se o resultado é um DataFrame com as colunas corretas
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(list(features.columns), self.mock_columns)
        self.assertEqual(features.shape, (1, len(self.mock_columns)))
    
    @patch('src.inference.load_model')
    @patch('src.inference.preprocess_url')
    def test_predict(self, mock_preprocess_url, mock_load_model):
        """Testa a função de predição."""
        from src.inference import predict
        
        # Configurar mocks
        mock_features = pd.DataFrame([[10, 5, 0, 1]], columns=self.mock_columns)
        mock_preprocess_url.return_value = mock_features
        
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]  # Predição: é phishing
        mock_load_model.return_value = (mock_model, self.mock_columns, None)
        
        # URL de teste
        test_url = "https://example.com"
        
        # Chamar a função
        result = predict(test_url)
        
        # Verificar se o modelo foi carregado
        mock_load_model.assert_called_once()
        
        # Verificar se a URL foi pré-processada
        mock_preprocess_url.assert_called_once_with(test_url)
        
        # Verificar se o modelo foi chamado para predição
        mock_model.predict.assert_called_once()
        
        # Verificar o resultado
        self.assertTrue(result)  # Deve ser True (phishing)
        
        # Testar com predição negativa
        mock_model.predict.return_value = [0]  # Predição: não é phishing
        result = predict(test_url)
        self.assertFalse(result)  # Deve ser False (não é phishing)


if __name__ == '__main__':
    unittest.main()
