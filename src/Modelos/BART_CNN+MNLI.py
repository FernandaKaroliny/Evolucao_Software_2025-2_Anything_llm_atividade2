import os
import re
import json
import unicodedata
import time
from transformers import pipeline
import subprocess
import tempfile
import shutil
import git
import stat
import os

# =========================
# CONFIGURAÇÕES
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URL_REPO = "https://github.com/Mintplex-Labs/anything-llm.git"
BRANCH = "master"
LIMITE_CARACTERES = 3000
ARQUIVO_SAIDA_TXT = os.path.join(BASE_DIR, "..", "..", "resultados", "bart_cnn+mnli.txt")

PADROES = [
    "Client-Server (a centralized server provides resources or services to multiple clients over a network)",
    "Blackboard (components work cooperatively by reading and writing shared data on a common knowledge base)",
    "Shared-Data (components communicate indirectly through shared data repositories or databases)",
    "Data-Model (the architecture centers around structured data schemas and access layers)",
    "Publish-Subscribe (components communicate asynchronously through message topics or events)",
    "Service-Oriented Architecture (system organized into reusable services communicating via standardized interfaces)",
    "Peer-to-Peer (decentralized network where each node can act as both client and server)",
    "Pipe-Filter (data flows through a sequence of processing steps, each transforming the input into output)",
    "Layers (system organized into hierarchical layers like presentation, logic, and data access)",
    "Microservices (independently deployable small services communicating via APIs or messaging)",
    "Blockchain (distributed ledger storing transactions in cryptographically linked blocks)"
]

# =========================
# BAIXAR REPOSITÓRIO
# =========================
# pasta um nível acima
repo_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "anything-llm"))

if not os.path.exists(repo_dir):
    print("📁 Repositório não encontrado. Clonando...")
    try:
        git.Repo.clone_from(URL_REPO, repo_dir, branch=BRANCH)
        print(f"✔️ Repositório clonado em: {repo_dir}")
    except Exception as e:
        print(f"❌ Erro ao clonar o repositório: {e}")
        exit()
else:
    print(f"📂 Repositório já existe em: {repo_dir}")
    print("✔️ Usando versão local.\n")


# Caminho usado no os.walk
CAMINHO_REPO = repo_dir

# =========================
# MODELOS
# =========================

print("🧠 Carregando modelos...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# =========================
# FUNÇÕES AUXILIARES
# =========================

def limpar_markdown(texto):
    texto = unicodedata.normalize("NFKD", texto)
    texto = re.sub(r"```.*?```", "", texto, flags=re.DOTALL)
    texto = re.sub(r"!\[.*?\]\(.*?\)", "", texto)
    texto = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", texto)
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"(^|\n)[#>\-\*\+]+\s*", "\n", texto)
    texto = re.sub(r"\n\s*\n+", "\n", texto)
    texto = re.sub(r" +", " ", texto)
    return texto.strip()

def dividir_texto(texto, limite):
    partes = []
    while len(texto) > limite:
        corte = texto[:limite].rfind(".")
        if corte == -1:
            corte = limite
        partes.append(texto[:corte])
        texto = texto[corte:]
    if texto.strip():
        partes.append(texto.strip())
    return partes

# =========================
# PROCESSAMENTO
# =========================

resultados = []
tempo_execucao = time.perf_counter()

for raiz, _, arquivos in os.walk(CAMINHO_REPO):
    for nome_arquivo in arquivos:
        if nome_arquivo.endswith(".md"):
            caminho = os.path.join(raiz, nome_arquivo)
            print(f"\n📄 Lendo {caminho}")

            with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
                conteudo = f.read()

            conteudo_limpo = limpar_markdown(conteudo)
            
            #with open("entradas_processadas.txt", "a", encoding="utf-8") as entrada_saida:
            #    entrada_saida.write(f"\n\n===== {caminho} =====\n")
            #    entrada_saida.write(conteudo_limpo)
            #    entrada_saida.write("\n")

            if not conteudo_limpo.strip():
                print("⚪ Ignorado (sem conteúdo relevante)")
                continue

            partes = dividir_texto(conteudo_limpo, LIMITE_CARACTERES)
            resumo_final = ""

            for i, parte in enumerate(partes):
                input_length = len(parte.split())
                max_len = max(40, int(input_length * 0.8))
                min_len = max(20, int(input_length * 0.3))

                try:
                    resumo = summarizer(
                        parte,
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=False
                    )[0]["summary_text"]
                    resumo_final += resumo + " "
                except Exception as e:
                    print(f"⚠️ Erro ao resumir parte {i+1}: {e}")

            if resumo_final.strip():
                try:
                    classificacao = classifier(
                        resumo_final,
                        candidate_labels=PADROES,
                        hypothesis_template="This project follows the following software architecture pattern: {}."
                    )
                    padrao_predito = classificacao["labels"][0]
                    confianca = classificacao["scores"][0]

                    print(f"🔹 {nome_arquivo}: {padrao_predito} ({confianca:.1%})")

                    resultados.append({
                        "arquivo": nome_arquivo,
                        "resumo": resumo_final.strip(),
                        "padrao_arquitetural": padrao_predito,
                        "confianca": round(confianca, 3)
                    })
                except Exception as e:
                    print(f"⚠️ Erro ao classificar {nome_arquivo}: {e}")

# =========================
# SALVAR RESULTADOS
# =========================

with open(ARQUIVO_SAIDA_TXT, "w", encoding="utf-8") as f:
    f.write("=== RESULTADOS DA CLASSIFICAÇÃO DE PADRÕES ARQUITETURAIS ===\n\n")

    for item in resultados:
        f.write(f"Arquivo: {item.get('arquivo', 'N/A')}\n")
        f.write(f"Padrão Arquitetural: {item.get('padrao_arquitetural', 'N/A')}\n")
        f.write(f"Confiança: {item.get('confianca', 0):.2%}\n")
        resumo = item.get('resumo', '').strip().replace("\n", " ")
        f.write(f"Resumo: {resumo}\n")
        f.write("-" * 60 + "\n")

    # Estatísticas
    from collections import Counter, defaultdict

    contagem_padroes = Counter()
    soma_confiancas = defaultdict(float)

    for item in resultados:
        padrao = item["padrao_arquitetural"]
        confianca = item["confianca"]
        contagem_padroes[padrao] += 1
        soma_confiancas[padrao] += confianca

    total_arquivos = len(resultados)
    padrao_mais_comum = contagem_padroes.most_common(1)[0]
    padrao_nome = padrao_mais_comum[0]
    padrao_qtd = padrao_mais_comum[1]
    padrao_confianca_media = soma_confiancas[padrao_nome] / padrao_qtd

    f.write("\n=== ESTATÍSTICAS GERAIS ===\n")
    f.write(f"Total de arquivos analisados: {total_arquivos}\n\n")
    f.write("Distribuição de padrões detectados:\n")

    for padrao, qtd in contagem_padroes.most_common():
        confianca_media = soma_confiancas[padrao] / qtd
        f.write(f" - {padrao}: {qtd} ocorrências (média {confianca_media:.1%})\n")

    f.write("\n=== PADRÃO MAIS PROVÁVEL ===\n")
    f.write(f"Padrão predominante: {padrao_nome}\n")
    f.write(f"Ocorrências: {padrao_qtd}\n")
    f.write(f"Confiança média: {padrao_confianca_media:.1%}\n")

print(f"📄 TXT salvo em: {ARQUIVO_SAIDA_TXT}")

tempo_execucao = time.perf_counter() - tempo_execucao    

print("\n✅ Análise concluída!")
print(f"📁 Resultados salvos em: {ARQUIVO_SAIDA_TXT} {ARQUIVO_SAIDA_TXT}")

print(f"⏱️ Tempo de execução {tempo_execucao}")

