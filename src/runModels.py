import os
import sys
import subprocess
import time
from git import Repo


repo_url = "https://github.com/Mintplex-Labs/anything-llm"
repo_dir = "anything-llm"
if not os.path.exists(repo_dir):
    print("Clonando repositório...")
    Repo.clone_from(repo_url, repo_dir)
    print("Clone concluído!\n")
else:
    print("Repositório já existe. Usando versão local.\n")


# ===========================================
# CONFIGURAÇÃO DOS SCRIPTS DOS MODELOS
# ===========================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Criar a pasta 'resultados' UM NÍVEL ACIMA DA SRC
RESULTADOS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "resultados"))
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Caminho base da pasta Modelos
MODELOS_DIR = os.path.join(BASE_DIR, "Modelos")

MODEL_SCRIPTS = {
    "1": ("BART (bart-large-cnn + bart-large-mnli)",
          os.path.join(MODELOS_DIR, "bart_cnn+mnli.py")),

    "2": ("LLaMA",
          os.path.join(MODELOS_DIR, "llama.py")),

    "3": ("Qwen",
          os.path.join(MODELOS_DIR, "qwen.py")),

    "4": ("Bart_MNLM",
          os.path.join(MODELOS_DIR, "bart_mnli.py")),

    "5": ("All-MiniLM-L6-v2",
          os.path.join(MODELOS_DIR, "all_minilm_l6_v2.py")),

    "0": ("Sair", None)
}

# ===========================================
# FUNÇÕES
# ===========================================

def limpar_terminal():
    os.system("cls" if os.name == "nt" else "clear")


def mostrar_menu():
    print("=======================================")
    print("        SELECIONE O MODELO LLM         ")
    print("=======================================\n")

    for key, (nome, _) in MODEL_SCRIPTS.items():
        print(f" {key} — {nome}")

    print("\n=======================================\n")


def executar_modelo(script):
    if not os.path.exists(script):
        print(f"\n❌ ERRO: O script '{script}' não foi encontrado!")
        return

    print(f"\n🚀 Executando modelo via: {script}\n")
    time.sleep(1)

    try:
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ O modelo encontrou um erro:\n{e}")
    except Exception as e:
        print(f"\n❌ Falha inesperada ao executar o modelo:\n{e}")


# ===========================================
# PROGRAMA PRINCIPAL
# ===========================================

def main():
    while True:
        limpar_terminal()
        mostrar_menu()
        escolha = input("👉 Digite o número da opção: ").strip()

        if escolha not in MODEL_SCRIPTS:
            print("\n⚠️ Opção inválida, tente novamente...")
            time.sleep(1.5)
            continue

        nome, script = MODEL_SCRIPTS[escolha]

        if escolha == "0":
            print("\n👋 Encerrando...")
            break

        print(f"\nModelo selecionado: {nome}\n")
        executar_modelo(script)

        input("\nPressione ENTER para voltar ao menu...")


if __name__ == "__main__":
    main()
