from transformers import pipeline
from git import Repo
from tqdm import tqdm
import shutil
import os

# ==========================================================
# 1. DEFINIÇÃO DO REPOSITÓRIO
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
repo_url = "https://github.com/Mintplex-Labs/anything-llm"
repo_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "anything-llm"))



# ==========================================================
# 2. CLONAR SOMENTE SE NÃO EXISTIR
# ==========================================================

if not os.path.exists(repo_dir):
    print("Repositório não encontrado. Clonando...")
    Repo.clone_from(repo_url, repo_dir)
    print("Clone concluído!\n")
else:
    print("Repositório já existe. Usando versão local.\n")

# ==========================================================
# 3. CARREGAR O MODELO BART-LARGE-MNLI
# ==========================================================

print("Carregando modelo bart-large-mnli...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("Modelo carregado!\n")

# ==========================================================
# 4. PADRÕES ARQUITETURAIS
# ==========================================================

architectural_patterns = {
    "MVC (Model-View-Controller)": "Separates software into Model, View, and Controller layers to organize logic and interface independently.",
    "MVVM (Model-View-ViewModel)": "Enhances separation between UI and logic through reactive data binding and ViewModel mediation.",
    "Clean Architecture": "Organizes the system in concentric layers isolating business rules from frameworks and external details.",
    "Hexagonal Architecture": "Builds systems around a domain core using ports and adapters to allow flexible integration layers.",
    "Layered Architecture": "Traditional N-layer approach where presentation, business, and data layers interact hierarchically.",
    "Microservices": "Application composed of independent services communicating via lightweight APIs.",
    "Event-Driven Architecture": "Components communicate by producing and reacting to asynchronous events.",
    "Monolithic Architecture": "Single deployable unit where all logic resides in one tightly integrated codebase."
}

pattern_labels = list(architectural_patterns.keys())
pattern_descriptions = list(architectural_patterns.values())

# ==========================================================
# 5. LEITURA SEGURA DE ARQUIVOS
# ==========================================================

def read_file_safe(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except:
        return ""

# ==========================================================
# 6. ARQUIVOS DE LOG
# ==========================================================


#input_log = "arquivos_lidos.txt"
#output_log = "resultados_padroes.txt"

entradas_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "entradas"))
ARQUIVO_ENTRADA_TXT = os.path.join(entradas_dir, "entrada2.txt")
ARQUIVO_SAIDA_TXT = os.path.join(BASE_DIR, "..", "..", "resultados", "Bart_MNLI.txt")

os.makedirs(entradas_dir, exist_ok=True)



#open(input_log, "w", encoding="utf-8").close()

open(ARQUIVO_SAIDA_TXT, "w", encoding="utf-8").close()

# ==========================================================
# 7. ANALISAR APENAS ARQUIVOS .JS
# ==========================================================

print("Analisando arquivos .js...\n")

for root, _, files in os.walk(repo_dir):
    for file in tqdm(files, desc="Arquivos JS"):
        if not file.endswith(".js"):
            continue

        path = os.path.join(root, file)
        code = read_file_safe(path)

        if not code.strip():
            continue

        # Registrar arquivo lido
        #with open(input_log, "a", encoding="utf-8") as f:
        #    f.write(path + "\n")

        # Limitar texto para classificação
        snippet = code[:2000]

        # Zero-shot classification
        result = classifier(
            snippet,
            pattern_descriptions,
            multi_label=True
        )

        # TOP 3 padrões
        top3 = sorted(
            zip(pattern_labels, result["scores"]),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        # Salvar no arquivo de saída
        with open(ARQUIVO_SAIDA_TXT, "a", encoding="utf-8") as f:
            f.write(f"\nArquivo: {path}\n")
            for label, score in top3:
                f.write(f"- {label}: {score:.4f}\n")

# ==========================================================
# FINALIZAÇÃO
# ==========================================================

print("\nProcesso concluído!")
#rint(f"Arquivos lidos salvos em: {input_log}")
print(f"Resultados salvos em: {ARQUIVO_SAIDA_TXT}")
