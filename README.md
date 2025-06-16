# Projeto de Detecção de Phishing

Este repositório contém o código e os recursos para um modelo de detecção de phishing baseado em aprendizado de máquina.

## Estrutura do Projeto

```
├── README.md                # Descrição geral do projeto
├── .gitignore               # Arquivos e pastas a serem ignorados pelo Git
├── docker-compose.yml       # Configuração do Docker Compose
├── Dockerfile               # Configuração do ambiente Docker
├── requirements.txt         # Dependências do projeto
├── notebooks/               # Notebooks Jupyter para estudo e experimentação
│   ├── 00_DatasetGeneration.ipynb
│   ├── 01_AutoMLImplementation.ipynb
│   └── ExploratoryAnalysis.ipynb
├── data/                    # Dados utilizados no projeto
│   ├── phishing_dataset_CIS.csv
│   ├── colunas_auxiliares.pkl
│   ├── colunas_treinamento.pkl
│   └── modelo_phishing_final.pkl
├── src/                     # Código-fonte do projeto
│   ├── __init__.py
│   ├── preprocessing.py     # Funções de pré-processamento
│   ├── training.py          # Código para treinamento do modelo
│   ├── inference.py         # Código para inferência com o modelo treinado
│   └── utils.py             # Funções auxiliares
├── tests/                   # Testes automatizados
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_inference.py
├── presentation/            # Materiais da apresentação
│   ├── slides.pdf
│   └── notes.md
└── docs/                    # Documentação adicional
    ├── challenges.md
    ├── trends.md
    └── references.md
```

## Sobre o Projeto

Este projeto implementa um sistema de detecção de phishing utilizando técnicas de aprendizado de máquina. O sistema analisa características de URLs para identificar potenciais ameaças de phishing.

## Requisitos

- Python 3.10+
- Pandas
- NumPy
- Scikit-learn
- PyCaret
- BeautifulSoup4
- Docker (opcional)

## Instalação e Execução

### Opção 1: Usando Docker

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/CIS-IEE-Project.git
cd CIS-IEE-Project
```

2. Execute o projeto com Docker Compose:
```bash
docker-compose up
```

3. Acesse o serviço de inferência em http://localhost:8000 ou o ambiente Jupyter em http://localhost:8888.

### Opção 2: Instalação Local

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/CIS-IEE-Project.git
cd CIS-IEE-Project
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Execute os notebooks ou scripts conforme necessário:
```bash
python -m src.inference  # para executar o serviço de inferência
jupyter notebook notebooks/  # para explorar os notebooks
```

## Como Usar o Modelo

Para utilizar o modelo de detecção de phishing em seu próprio código:

```python
from src.inference import predict

# URL a ser analisada
url = "http://exemplo.com"

# Realizar a predição
result = predict(url)
print(f"A URL {url} {'é' if result else 'não é'} phishing.")
```

## Contribuições

Contribuições são bem-vindas! Por favor, sinta-se à vontade para enviar pull requests ou abrir issues se encontrar problemas.

## Licença

[Especifique a licença aqui]