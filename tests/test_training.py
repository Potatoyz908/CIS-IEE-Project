"""
Testes para o módulo de treinamento.
"""

import unittest
import os
import pandas as pd
import pickle
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestTraining(unittest.TestCase):
    """Testes unitários para as funções de treinamento."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Criar um DataFrame de teste
        self.test_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature3': [True, False, True, False, True],
            'phishing': [1, 0, 1, 0, 1]
        })
    
    @patch('src.training.pd.read_csv')
    def test_load_data(self, mock_read_csv):
        """Testa o carregamento de dados."""
        from src.training import load_data, DATASET_PATH
        
        # Configurar o mock para retornar o DataFrame de teste
        mock_read_csv.return_value = self.test_df
        
        # Chamar a função
        df = load_data()
        
        # Verificar se read_csv foi chamado com o caminho correto
        mock_read_csv.assert_called_once_with(DATASET_PATH)
        
        # Verificar se o DataFrame retornado é o esperado
        self.assertEqual(df.shape, self.test_df.shape)
        self.assertTrue(all(df.columns == self.test_df.columns))
    
    @patch('src.training.pickle.dump')
    def test_preprocess_training_data(self, mock_pickle_dump):
        """Testa o pré-processamento de dados para treinamento."""
        from src.training import preprocess_training_data
        
        # Chamar a função
        X, y = preprocess_training_data(self.test_df)
        
        # Verificar se X e y têm as dimensões corretas
        self.assertEqual(X.shape, (5, 3))  # 5 linhas, 3 colunas (excluindo 'phishing')
        self.assertEqual(y.shape, (5,))    # 5 valores
        
        # Verificar se as colunas em X não incluem 'phishing'
        self.assertNotIn('phishing', X.columns)
        
        # Verificar se pickle.dump foi chamado para salvar as colunas
        mock_pickle_dump.assert_called_once()
        
        # Verificar os valores de y
        self.assertTrue(all(y == self.test_df['phishing']))
    
    @patch('src.training.compare_models')
    @patch('src.training.tune_model')
    @patch('src.training.predict_model')
    @patch('src.training.finalize_model')
    @patch('src.training.pickle.dump')
    @patch('src.training.setup')
    def test_train_automl_model(self, mock_setup, mock_pickle_dump, 
                                mock_finalize_model, mock_predict_model, 
                                mock_tune_model, mock_compare_models):
        """Testa o treinamento do modelo usando AutoML."""
        from src.training import train_automl_model
        
        # Configurar mocks
        mock_best_model = MagicMock()
        mock_tuned_model = MagicMock()
        mock_final_model = MagicMock()
        mock_predictions = pd.DataFrame({
            'prediction_label': [1, 0, 1, 0, 1]
        })
        
        mock_compare_models.return_value = mock_best_model
        mock_tune_model.return_value = mock_tuned_model
        mock_predict_model.return_value = mock_predictions
        mock_finalize_model.return_value = mock_final_model
        
        # Separar X e y do DataFrame de teste
        X = self.test_df.drop('phishing', axis=1)
        y = self.test_df['phishing']
        
        # Chamar a função
        model = train_automl_model(X, y)
        
        # Verificar se os métodos do PyCaret foram chamados na ordem correta
        mock_setup.assert_called_once()
        mock_compare_models.assert_called_once()
        mock_tune_model.assert_called_once_with(mock_best_model, optimize='f1')
        mock_predict_model.assert_called_once()
        mock_finalize_model.assert_called_once_with(mock_tuned_model)
        
        # Verificar se o modelo final foi salvo
        mock_pickle_dump.assert_called_once_with(mock_final_model, mock_pickle_dump.call_args[0][1])
        
        # Verificar se o modelo retornado é o final
        self.assertEqual(model, mock_final_model)


if __name__ == '__main__':
    unittest.main()
