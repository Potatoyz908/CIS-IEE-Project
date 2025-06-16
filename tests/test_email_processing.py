"""
Testes para o módulo de processamento de e-mails.
"""

import unittest
from src.email_processing import EmailProcessor


class TestEmailProcessing(unittest.TestCase):
    """Testes unitários para o processador de e-mails."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        self.processor = EmailProcessor()
    
    def test_parse_email(self):
        """Testa a análise de e-mails."""
        # Gerar um e-mail de teste (phishing)
        test_email = self.processor.generate_test_email(is_phishing=True)
        
        # Parsear o e-mail
        parsed_email = self.processor.parse_email(test_email)
        
        # Verificar campos básicos
        self.assertEqual(parsed_email['subject'], "URGENT: Your Account Access Will Be Suspended")
        self.assertTrue("Security Team" in parsed_email['from'])
        self.assertEqual(parsed_email['to'], "User <user@example.com>")
        
        # Verificar extração de URLs
        self.assertEqual(len(parsed_email['urls']), 1)
        self.assertEqual(parsed_email['urls'][0], "http://malicious-site.com/verify.php")
        
        # Verificar conteúdo do corpo
        self.assertTrue("unusual activity" in parsed_email['body_text'])
        self.assertTrue("<b>unusual activity</b>" in parsed_email['body_html'])
    
    def test_extract_email_features(self):
        """Testa a extração de características de e-mails."""
        # Gerar e parsear um e-mail de phishing
        phishing_email = self.processor.generate_test_email(is_phishing=True)
        parsed_phishing = self.processor.parse_email(phishing_email)
        
        # Extrair características
        phishing_features = self.processor.extract_email_features(parsed_phishing)
        
        # Verificar características específicas de phishing
        self.assertEqual(phishing_features['subject_has_urgent'], 1)
        self.assertTrue(phishing_features['phishing_keyword_count'] > 0)
        
        # Gerar e parsear um e-mail legítimo
        legitimate_email = self.processor.generate_test_email(is_phishing=False)
        parsed_legitimate = self.processor.parse_email(legitimate_email)
        
        # Extrair características
        legitimate_features = self.processor.extract_email_features(parsed_legitimate)
        
        # Verificar características específicas de e-mail legítimo
        self.assertEqual(legitimate_features['subject_has_urgent'], 0)
        self.assertTrue(legitimate_features['phishing_keyword_count'] < phishing_features['phishing_keyword_count'])
    
    def test_preprocess_text(self):
        """Testa o pré-processamento de texto."""
        test_text = "This is an Example TEXT with Numbers 12345 and Special Characters!@#"
        processed = self.processor._preprocess_text(test_text)
        
        # Verificar se está em minúsculas
        self.assertFalse(any(c.isupper() for c in processed))
        
        # Verificar se removeu números
        self.assertFalse(any(c.isdigit() for c in processed))
        
        # Verificar se removeu caracteres especiais
        self.assertFalse(any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in processed))
    
    def test_generate_test_email(self):
        """Testa a geração de e-mails de teste."""
        # Gerar um e-mail de phishing
        phishing = self.processor.generate_test_email(is_phishing=True)
        self.assertTrue("URGENT" in phishing)
        self.assertTrue("secure-banking-verification.com" in phishing)
        
        # Gerar um e-mail legítimo
        legitimate = self.processor.generate_test_email(is_phishing=False)
        self.assertTrue("Team Meeting" in legitimate)
        self.assertTrue("company.com" in legitimate)
    
    def test_preprocess_emails_for_model(self):
        """Testa o pré-processamento de múltiplos e-mails."""
        # Gerar dois e-mails
        emails = [
            self.processor.generate_test_email(is_phishing=True),
            self.processor.generate_test_email(is_phishing=False)
        ]
        
        # Pré-processar
        df = self.processor.preprocess_emails_for_model(emails)
        
        # Verificar resultado
        self.assertEqual(len(df), 2)
        self.assertTrue('subject_length' in df.columns)
        self.assertTrue('phishing_keyword_count' in df.columns)
        
        # Verificar se colunas não numéricas foram removidas
        self.assertFalse('sender_domain' in df.columns)
        self.assertFalse('processed_text' in df.columns)


if __name__ == '__main__':
    unittest.main()
