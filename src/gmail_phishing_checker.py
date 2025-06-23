import os
import base64
from datetime import datetime
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import pandas as pd
import joblib
import nltk

from sentence_transformers import SentenceTransformer
from src.email_processing import EmailProcessor
from src.preprocessing import preprocess_data, prepare_for_model

# Baixar recurso necessário do NLT
nltk.download('punkt_tab')
nltk.download('punkt')

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def main():
    # Carregar modelo e variáveis
    final_model = joblib.load('data/modelo_phishing_final.pkl')
    model_columns_url = joblib.load('data/colunas_treinamento.pkl')  # <- agora usando o .pkl correto
    aux_columns = joblib.load('data/colunas_auxiliares.pkl')         # <- idem
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    creds = authenticate()
    service = build('gmail', 'v1', credentials=creds)

    print("🔎 Buscando e-mails não lidos...")
    results = service.users().messages().list(userId='me', labelIds=['INBOX', 'UNREAD'], maxResults=10).execute()
    messages = results.get('messages', [])
    print(f"📬 {len(messages)} e-mail(s) novo(s) encontrado(s).")

    if not messages:
        return

    email_processor = EmailProcessor()

    for msg in messages:
        msg_raw = service.users().messages().get(userId='me', id=msg['id'], format='raw').execute()
        raw_bytes = base64.urlsafe_b64decode(msg_raw['raw'].encode('ASCII'))

        parsed_email = email_processor.parse_email(raw_bytes)
        email_features = email_processor.extract_email_features(parsed_email)

        # Extração de horário e período
        timestamp = int(msg_raw['internalDate']) // 1000
        hora_recebimento = datetime.fromtimestamp(timestamp)
        hora = hora_recebimento.strftime("%H:%M")
        h = hora_recebimento.hour
        if 5 <= h < 12:
            periodo = "manha"
        elif 12 <= h < 18:
            periodo = "tarde"
        elif 18 <= h < 24:
            periodo = "noite"
        else:
            periodo = "madrugada"

        # Montagem das features no padrão isItPhishing
        features_for_model = {
            'time': h,
            'urls': int(bool(parsed_email.get('urls'))),
            'sendingPeriod': periodo,
            'subject': parsed_email.get('subject', ''),
            'subjectClear': email_features.get('processed_text', ''),  # campo processado
            'body': parsed_email.get('body_text', ''),
            'bodyClear': email_features.get('processed_text', '')      # mesmo processamento usado
        }

        # Embedding do texto processado
        email_text = features_for_model['subjectClear'] + ' ' + features_for_model['bodyClear']
        embedding = sbert.encode([email_text])
        embedding_df = pd.DataFrame(embedding, columns=[f'emb_{i}' for i in range(embedding.shape[1])])

        # Features auxiliares (time, urls, sendingPeriod dummificado)
        X_aux = pd.DataFrame([{
            'time': features_for_model['time'],
            'urls': features_for_model['urls'],
            'sendingPeriod': features_for_model['sendingPeriod']
        }])

        # Dummies para 'sendingPeriod'
        X_aux = pd.get_dummies(X_aux, columns=['sendingPeriod'], drop_first=False)

        # Garantir colunas auxiliares compatíveis
        for col in aux_columns:
            if col not in X_aux.columns:
                X_aux[col] = 0
        X_aux = X_aux[aux_columns]  # garantir ordem

        # Juntar tudo (embeddings + variáveis auxiliares)
        input_final = pd.concat([embedding_df.reset_index(drop=True), X_aux.reset_index(drop=True)], axis=1)

        # Garantir colunas do modelo
        for col in model_columns_url:
            if col not in input_final.columns:
                input_final[col] = 0  # adiciona feature faltante

        # Remover 'phishing' caso tenha entrado por acidente
        if 'phishing' in input_final.columns:
            input_final = input_final.drop(columns=['phishing'])

        input_final = input_final[model_columns_url]  # ordem correta

        # Predição final
        phishing_prob = final_model.predict_proba(input_final)[0, 1]

        print("\n==============================")
        print(f"🕒 Horário: {hora} ({periodo})")
        print(f"📨 Assunto: {parsed_email['subject']}")
        print(f"⚠️ Probabilidade de phishing: {phishing_prob * 100:.2f}%")

if __name__ == '__main__':
    main()
