import re
import unicodedata
import time
import os
from transformers import pipeline, set_seed

# =========================
# REPRODUTIBILIDADE
# =========================
set_seed(42)

# =========================
# CONFIGURAÇÕES
# =========================
ARQUIVO_ENTRADA = "Evolucao_Software_2025-2_Anything_llm_atividade2/entradas/entrada6.txt"
ARQUIVO_SAIDA_TXT = "Evolucao_Software_2025-2_Anything_llm_atividade2/resultados/bart_mnli+entrada_reduzida.txt"

MAX_RESUMO_CHARS = 8000

# =========================
# RÓTULOS DE CLASSIFICAÇÃO
# =========================

PADROES_RELEASE = [
    "Continuous Deployment (every code change that passes automated tests is automatically deployed to production without manual approval)",
    "Continuous Delivery (code is continuously integrated and tested, remaining ready for deployment at any time, but releases require a manual decision)",
    "Scheduled Releases (software is released at predefined time intervals, such as weekly, monthly, or at fixed calendar dates)",
    "Big Bang Release (a large set of features is released simultaneously in a single major deployment event)",
    "Canary Release (a new version is gradually rolled out to a small subset of users to monitor behavior before full deployment)",
    "Blue-Green Deployment (two identical production environments are maintained, allowing instant traffic switch between old and new versions)",
    "Rolling Release (updates are deployed incrementally across servers or users without downtime, replacing old versions progressively)"
]

PADROES_FLUXO = [
    "Git Flow (uses multiple long-lived branches such as main, develop, feature, release, and hotfix to manage parallel development and releases)", 
    "GitHub Flow (a lightweight workflow based on a single main branch with short-lived feature branches and pull requests)",
    "GitLab Flow (extends GitHub Flow by introducing environment-based or release-based branching strategies)", 
    "Trunk-Based Development (developers commit small, frequent changes directly to a shared main branch, relying on feature flags and continuous integration)",
    "Feature Branch Workflow (each new feature or change is developed in an isolated branch and merged into the main branch after review)"
]

# =========================
# MODELO (APENAS MNLI)
# =========================

print("🧠 Carregando modelo BART-MNLI...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# =========================
# FUNÇÕES AUXILIARES
# =========================

def limpar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)
    texto = re.sub(r"```.*?```", "", texto, flags=re.DOTALL)
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"\n\s*\n+", "\n", texto)
    texto = re.sub(r" +", " ", texto)
    return texto.strip()

def gerar_resumo_simples(texto: str, limite: int) -> str:
    """
    Resumo determinístico baseado no próprio texto.
    Evita uso de modelo de sumarização.
    """
    if len(texto) <= limite:
        return texto
    return texto[:limite].rsplit(".", 1)[0] + "."

def gerar_ranking(resultado):
    return [
        {
            "posicao": i + 1,
            "padrao": label,
            "confianca": score
        }
        for i, (label, score) in enumerate(
            zip(resultado["labels"], resultado["scores"])
        )
    ]

# =========================
# PIPELINE PRINCIPAL
# =========================

inicio = time.perf_counter()

with open(ARQUIVO_ENTRADA, "r", encoding="utf-8") as f:
    texto = limpar_texto(f.read())

if not texto:
    raise ValueError("❌ A entrada está vazia.")

resumo = gerar_resumo_simples(texto, MAX_RESUMO_CHARS)

# =========================
# CLASSIFICAÇÃO
# =========================

print("🏷️ Classificando Estratégia de Release...")
class_release = classifier(
    texto,
    candidate_labels=PADROES_RELEASE,
    hypothesis_template="The software project follows this release strategy: {}."
)

print("🔁 Classificando Fluxo de Trabalho...")
class_fluxo = classifier(
    texto,
    candidate_labels=PADROES_FLUXO,
    hypothesis_template="The development workflow follows: {}."
)

fim = time.perf_counter()

ranking_release = gerar_ranking(class_release)
ranking_fluxo = gerar_ranking(class_fluxo)

# =========================
# SALVAR RESULTADOS
# =========================

with open(ARQUIVO_SAIDA_TXT, "w", encoding="utf-8") as f:
    f.write("=== CLASSIFICAÇÃO  ===\n\n")

    f.write("📥 RESUMO DO TEXTO ANALISADO\n")
    f.write(resumo + "\n\n")

    f.write("📦 ESTRATÉGIA DE RELEASE\n")
    f.write(f"Resultado principal: {ranking_release[0]['padrao']}\n")
    f.write(f"Confiança: {ranking_release[0]['confianca']:.2%}\n\n")
    f.write("Ranking completo:\n")
    for r in ranking_release:
        f.write(f" {r['posicao']}º {r['padrao']} — {r['confianca']:.1%}\n")

    f.write("\n🔁 FLUXO DE TRABALHO\n")
    f.write(f"Resultado principal: {ranking_fluxo[0]['padrao']}\n")
    f.write(f"Confiança: {ranking_fluxo[0]['confianca']:.2%}\n\n")
    f.write("Ranking completo:\n")
    for r in ranking_fluxo:
        f.write(f" {r['posicao']}º {r['padrao']} — {r['confianca']:.1%}\n")

    f.write(f"\n⏱️ Tempo total: {fim - inicio:.2f} segundos\n")

print(f"\n📄 Resultado salvo em: {os.path.abspath(ARQUIVO_SAIDA_TXT)}")
print(f"⏱️ Tempo de execução: {fim - inicio:.2f}s")
