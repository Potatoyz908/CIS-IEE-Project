"""
Módulo de processamento de e-mails para detecção de phishing.
Este módulo fornece funcionalidades para processar e-mails e extrair features para análise de phishing.
"""

import re
import email
import base64
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from email import policy
from email.header import decode_header
from email.parser import BytesParser, Parser
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Baixar recursos do NLTK se necessário
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class EmailProcessor:
    """Classe para processar e-mails e extrair características para detecção de phishing."""
    
    def __init__(self):
        """Inicializa o processador de e-mails."""
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Palavras comumente usadas em e-mails de phishing
        self.phishing_keywords = [
            'account', 'update', 'security', 'bank', 'verify', 'login', 'confirm',
            'password', 'credit', 'click', 'urgent', 'alert', 'suspend', 'restrict',
            'unauthorized', 'limited', 'access', 'information', 'identity', 'time'
        ]
    
    def parse_email(self, email_content):
        """
        Analisa o conteúdo do e-mail para extrair componentes importantes.
        
        Args:
            email_content: String contendo o conteúdo bruto do e-mail ou objeto EmailMessage
                          Pode ser o conteúdo de um arquivo .eml ou .msg
            
        Returns:
            Dicionário com os componentes extraídos do e-mail
        """
        # Se já for um objeto EmailMessage, retorna diretamente
        if hasattr(email_content, 'get'):
            email_message = email_content
        
        # Se for bytes, usar BytesParser
        elif isinstance(email_content, bytes):
            parser = BytesParser(policy=policy.default)
            email_message = parser.parsebytes(email_content)
        
        # Se for string, usar Parser
        elif isinstance(email_content, str):
            parser = Parser(policy=policy.default)
            email_message = parser.parsestr(email_content)
        
        else:
            raise ValueError("Tipo de entrada inválido para parse_email")

        
        # Extrair informações básicas
        parsed_email = {
            'subject': self._decode_field(email_message.get('Subject', '')),
            'from': self._decode_field(email_message.get('From', '')),
            'to': self._decode_field(email_message.get('To', '')),
            'date': email_message.get('Date', ''),
            'body_text': '',
            'body_html': '',
            'urls': [],
            'has_attachments': False,
            'attachments': []
        }
        
        # Processar corpo do e-mail
        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                content_disposition = part.get('Content-Disposition', '')
                
                # Verificar se é um anexo
                if 'attachment' in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        parsed_email['has_attachments'] = True
                        parsed_email['attachments'].append(filename)
                    continue
                
                # Extrair texto do corpo
                if content_type == 'text/plain':
                    body = part.get_payload(decode=True)
                    charset = part.get_content_charset()
                    if charset:
                        body = body.decode(charset, errors='replace')
                    else:
                        body = body.decode('utf-8', errors='replace')
                    parsed_email['body_text'] += body
                
                # Extrair HTML do corpo
                elif content_type == 'text/html':
                    body = part.get_payload(decode=True)
                    charset = part.get_content_charset()
                    if charset:
                        body = body.decode(charset, errors='replace')
                    else:
                        body = body.decode('utf-8', errors='replace')
                    parsed_email['body_html'] += body
                    
                    # Extrair URLs do HTML
                    soup = BeautifulSoup(body, 'html.parser')
                    for link in soup.find_all('a'):
                        url = link.get('href')
                        if url:
                            parsed_email['urls'].append(url)
        else:
            # E-mail simples, não multipart
            content_type = email_message.get_content_type()
            body = email_message.get_payload(decode=True)
            
            if body:
                charset = email_message.get_content_charset()
                if charset:
                    body = body.decode(charset, errors='replace')
                else:
                    body = body.decode('utf-8', errors='replace')
                
                if content_type == 'text/plain':
                    parsed_email['body_text'] = body
                elif content_type == 'text/html':
                    parsed_email['body_html'] = body
                    
                    # Extrair URLs do HTML
                    soup = BeautifulSoup(body, 'html.parser')
                    for link in soup.find_all('a'):
                        url = link.get('href')
                        if url:
                            parsed_email['urls'].append(url)
        
        # Se temos HTML mas não temos texto, extrair texto do HTML
        if not parsed_email['body_text'] and parsed_email['body_html']:
            soup = BeautifulSoup(parsed_email['body_html'], 'html.parser')
            parsed_email['body_text'] = soup.get_text()
        
        # Extrair URLs do texto
        if parsed_email['body_text']:
            url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+|https?%3A%2F%2F[^\s<>"]+|http%3A%2F%2F[^\s<>"]+|hxxps?://[^\s<>"]+|hxxp://[^\s<>"]+|h+t+p+s?://[^\s<>"]+|h+t+p://[^\s<>"]+'
            urls = re.findall(url_pattern, parsed_email['body_text'])
            for url in urls:
                if url not in parsed_email['urls']:
                    parsed_email['urls'].append(url)
        
        return parsed_email
    
    def extract_email_features(self, parsed_email):
        """
        Extrai características do e-mail para detecção de phishing.
        
        Args:
            parsed_email: Dicionário com os componentes extraídos do e-mail
            
        Returns:
            Dicionário com as features extraídas
        """
        features = {}
        
        # Características do assunto
        subject = parsed_email['subject']
        features['subject_length'] = len(subject)
        features['subject_has_urgent'] = int(bool(re.search(r'urgent|immediate|alert|important|attention|critical|now', subject.lower())))
        features['subject_has_account'] = int(bool(re.search(r'account|password|login|access|security|verify|confirm', subject.lower())))
        features['subject_has_money'] = int(bool(re.search(r'money|cash|credit|bank|financial|fund|payment', subject.lower())))
        
        # Características do remetente
        sender = parsed_email['from']
        features['sender_has_name'] = int('<' in sender and '>' in sender)
        if '@' in sender:
            domain = sender.split('@')[-1].split('>')[0]
            features['sender_domain'] = domain
            features['sender_domain_length'] = len(domain)
            features['sender_is_free_email'] = int(bool(re.search(r'gmail\.com|yahoo\.com|hotmail\.com|outlook\.com|aol\.com', domain.lower())))
        else:
            features['sender_domain'] = ''
            features['sender_domain_length'] = 0
            features['sender_is_free_email'] = 0
        
        # Características do corpo
        body = parsed_email['body_text']
        features['body_length'] = len(body)
        features['body_word_count'] = len(body.split())
        
        # Processamento do texto
        processed_text = self._preprocess_text(body)
        features['processed_text'] = processed_text
        
        # Contagem de palavras-chave de phishing
        phishing_word_count = sum(1 for word in processed_text.split() if word in self.phishing_keywords)
        features['phishing_keyword_count'] = phishing_word_count
        
        # Características de URLs
        urls = parsed_email['urls']
        features['url_count'] = len(urls)
        
        if urls:
            # Características da primeira URL (geralmente a principal)
            main_url = urls[0]
            try:
                parsed_url = urlparse(main_url)
                features['main_url_domain'] = parsed_url.netloc
                features['main_url_path_length'] = len(parsed_url.path)
                features['main_url_is_https'] = int(parsed_url.scheme == 'https')
            except:
                features['main_url_domain'] = ''
                features['main_url_path_length'] = 0
                features['main_url_is_https'] = 0
            
            # Verificar discrepância entre texto e URL
            if parsed_email['body_html']:
                soup = BeautifulSoup(parsed_email['body_html'], 'html.parser')
                mismatched_links = 0
                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    text = link.text.strip()
                    if href and text and not text.startswith('http') and text not in href:
                        if not (text == 'aqui' or text == 'here' or text == 'click' or text == 'link'):
                            mismatched_links += 1
                features['mismatched_links'] = mismatched_links
            else:
                features['mismatched_links'] = 0
        else:
            features['main_url_domain'] = ''
            features['main_url_path_length'] = 0
            features['main_url_is_https'] = 0
            features['mismatched_links'] = 0
        
        # Características de anexos
        features['has_attachments'] = int(parsed_email['has_attachments'])
        features['attachment_count'] = len(parsed_email['attachments'])
        
        # Características de estilo
        features['body_has_html'] = int(bool(parsed_email['body_html']))
        if parsed_email['body_html']:
            features['html_length'] = len(parsed_email['body_html'])
        else:
            features['html_length'] = 0
        
        # Adicionar campos brutos para uso posterior (ex: embeddings SBERT)
        features['subject'] = parsed_email.get('subject', '')
        features['body_text'] = parsed_email.get('body_text', '')

        return features
    
    def _decode_field(self, field):
        """Decodifica campos do cabeçalho de e-mail."""
        if not field:
            return ""
            
        decoded_parts = []
        parts = decode_header(field)
        
        for part, encoding in parts:
            if isinstance(part, bytes):
                if encoding:
                    try:
                        decoded_parts.append(part.decode(encoding))
                    except:
                        decoded_parts.append(part.decode('utf-8', errors='replace'))
                else:
                    decoded_parts.append(part.decode('utf-8', errors='replace'))
            else:
                decoded_parts.append(part)
        
        return ''.join(decoded_parts)
    
    def _preprocess_text(self, text):
        """
        Pré-processa o texto para análise:
        - Remove caracteres especiais
        - Converte para minúsculas
        - Remove stopwords
        - Aplica stemming (radicialização de palavras)
        """
        # Converter para minúsculas
        text = text.lower()
        
        # Remover caracteres especiais e números
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenização
        tokens = word_tokenize(text)
        
        # Remover stopwords
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        
        # Aplicar stemming
        stemmed_tokens = [self.stemmer.stem(word) for word in filtered_tokens]
        
        # Unir tokens novamente
        processed_text = ' '.join(stemmed_tokens)
        
        return processed_text
    
    def generate_test_email(self, is_phishing=True):
        """
        Gera um e-mail de teste para fins de desenvolvimento.
        
        Args:
            is_phishing: Se True, gera um e-mail de phishing. Se False, gera um e-mail legítimo.
            
        Returns:
            String contendo um e-mail bruto para teste
        """
        if is_phishing:
            # Gerar e-mail de phishing
            from_name = "Security Team"
            from_domain = "secure-banking-verification.com"
            subject = "URGENT: Your Account Access Will Be Suspended"
            
            html_body = """
            <html>
            <body>
                <p>Dear Valued Customer,</p>
                <p>We have detected <b>unusual activity</b> on your account. Your account access will be <b>suspended</b> within 24 hours unless you verify your information immediately.</p>
                <p>Please <a href="http://malicious-site.com/verify.php">click here</a> to verify your account information.</p>
                <p>Failure to verify your information will result in account suspension.</p>
                <p>Security Department<br>Customer Service</p>
            </body>
            </html>
            """
            
            text_body = """
            Dear Valued Customer,
            
            We have detected unusual activity on your account. Your account access will be suspended within 24 hours unless you verify your information immediately.
            
            Please visit: http://malicious-site.com/verify.php to verify your account information.
            
            Failure to verify your information will result in account suspension.
            
            Security Department
            Customer Service
            """
        else:
            # Gerar e-mail legítimo
            from_name = "Alex Johnson"
            from_domain = "company.com"
            subject = "Team Meeting - Thursday, 10 AM"
            
            html_body = """
            <html>
            <body>
                <p>Hi team,</p>
                <p>Just a reminder that we have our weekly team meeting this Thursday at 10 AM in the conference room.</p>
                <p>Please <a href="https://company.com/meetings/agenda">check the agenda</a> and come prepared with your updates.</p>
                <p>Best regards,<br>Alex</p>
            </body>
            </html>
            """
            
            text_body = """
            Hi team,
            
            Just a reminder that we have our weekly team meeting this Thursday at 10 AM in the conference room.
            
            Please check the agenda at: https://company.com/meetings/agenda and come prepared with your updates.
            
            Best regards,
            Alex
            """
        
        # Criar e-mail no formato .eml
        email_content = f"""From: "{from_name}" <info@{from_domain}>
To: "User" <user@example.com>
Subject: {subject}
Date: Mon, 16 Jun 2025 10:30:45 -0500
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="boundary-string"

--boundary-string
Content-Type: text/plain; charset="utf-8"
Content-Transfer-Encoding: quoted-printable

{text_body}

--boundary-string
Content-Type: text/html; charset="utf-8"
Content-Transfer-Encoding: quoted-printable

{html_body}

--boundary-string--
"""
        return email_content
    
    def preprocess_emails_for_model(self, emails):
        """
        Pré-processa múltiplos e-mails para uso no modelo.
        
        Args:
            emails: Lista de strings contendo e-mails brutos
            
        Returns:
            DataFrame com as features extraídas
        """
        features_list = []
        
        for email_content in emails:
            # Parsear e-mail
            parsed_email = self.parse_email(email_content)
            
            # Extrair características
            features = self.extract_email_features(parsed_email)
            
            # Adicionar ao list
            features_list.append(features)
        
        # Converter para DataFrame
        df = pd.DataFrame(features_list)
        
        # Remover colunas não numéricas
        non_feature_cols = ['sender_domain', 'main_url_domain', 'processed_text']
        for col in non_feature_cols:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        return df