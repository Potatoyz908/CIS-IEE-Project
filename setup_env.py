import os
from getpass import getpass

# Solicita as informações ao usuário
email_address = input("Digite seu e-mail de origem (EMAIL_ADDRESS): ")
email_password = getpass("Digite sua senha do e-mail (EMAIL_PASSWORD): ")
email_destino = input("Digite o e-mail de destino (EMAIL_DESTINO): ")

# Cria conteúdo do .env
env_content = (
    f"EMAIL_ADDRESS={email_address}\n"
    f"EMAIL_PASSWORD={email_password}\n"
    f"EMAIL_DESTINO={email_destino}\n"
)

# Escreve o .env
with open(".env", "w") as env_file:
    env_file.write(env_content)

print(".env criado com sucesso!")

# Adiciona .env ao .gitignore, se necessário
if os.path.exists(".gitignore"):
    with open(".gitignore", "r") as f:
        lines = f.read().splitlines()

    if ".env" not in lines:
        with open(".gitignore", "a") as f:
            f.write("\n.env\n")
        print(".env adicionado ao .gitignore.")
else:
    with open(".gitignore", "w") as f:
        f.write(".env\n")
    print(".gitignore criado e .env adicionado.")