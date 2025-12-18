# -*- coding: utf-8 -*-

import os
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

print("GPU disponível?", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("⚠️ GPU não disponível. Ative uma GPU ou use CUDA.")


# =========================
# CONFIGURAÇÃO DO MODELO
# =========================

model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

print(f"Carregando {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="cuda"
)


# =========================
# FUNÇÕES AUXILIARES
# =========================

def gerar_arvore_diretorios(caminho_raiz, max_depth=3, ignorar=['.git', 'node_modules', 'dist', 'build', 'coverage', 'venv', '.github', 'assets']):
    tree_str = ""
    root_level = caminho_raiz.count(os.sep)

    for root, dirs, files in os.walk(caminho_raiz):
        dirs[:] = [d for d in dirs if d not in ignorar]
        level = root.count(os.sep) - root_level
        if level > max_depth:
            continue

        indent = ' ' * 4 * level
        tree_str += f"{indent}{os.path.basename(root)}/\n"

        if level < max_depth:
            subindent = ' ' * 4 * (level + 1)
            for f in files[:10]:
                tree_str += f"{subindent}{f}\n"
            if len(files) > 10:
                tree_str += f"{subindent}... (+{len(files)-10} arquivos)\n"

    return tree_str


def extrair_resumo_readmes(repo_path, max_chars=2000):
    readmes_encontrados = ""
    locais_chave = [
        repo_path,
        os.path.join(repo_path, "server"),
        os.path.join(repo_path, "frontend"),
        os.path.join(repo_path, "collector")
    ]

    for pasta in locais_chave:
        caminho_arquivo = os.path.join(pasta, "README.md")

        if os.path.exists(caminho_arquivo):
            try:
                with open(caminho_arquivo, "r", encoding="utf-8", errors="ignore") as f:
                    conteudo = f.read()
                    conteudo = "\n".join([line for line in conteudo.splitlines() if line.strip()])
                    resumo = conteudo[:max_chars]

                    nome_pasta = (
                        "RAIZ DO PROJETO"
                        if os.path.basename(pasta) == os.path.basename(repo_path)
                        else os.path.basename(pasta)
                    )

                    readmes_encontrados += f"\n--- CONTEÚDO DO README ({nome_pasta}) ---\n"
                    readmes_encontrados += resumo
                    readmes_encontrados += "\n... (conteúdo truncado)...\n"
            except Exception as e:
                print(f"Erro ao ler {caminho_arquivo}: {e}")

    return readmes_encontrados


def inferir(texto):
    messages = [
        {"role": "system", "content": "Você é especialista em arquitetura de software."},
        {"role": "user", "content": texto}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.2
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("assistant")[-1].strip()


# =========================
# CARREGAR ESTRUTURA
# =========================

repo_path = "./anything-llm"

tree_visual = gerar_arvore_diretorios(repo_path)
readmes_content = extrair_resumo_readmes(repo_path)

print("\n=== TREE CAPTURADA (parcial) ===")
print("\n".join(tree_visual.splitlines()[:10]))
print("...")


# =========================
# CONFIGURAR PASTAS (entradas/ e resultados/)
# =========================

base_dir = os.path.dirname(os.path.abspath(__file__))

entradas_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "entradas"))
respostas_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "resultados"))

os.makedirs(entradas_dir, exist_ok=True)
os.makedirs(respostas_dir, exist_ok=True)


output_path = os.path.join(respostas_dir, "Qwen.txt")
input_path = os.path.join(entradas_dir, "entrada1.txt")

# Ler e imprimir o conteúdo existente do arquivo
print("\n===== CONTEÚDO DE entrada1.txt =====")
try:
    with open(input_path, "r", encoding="utf-8") as f:
        entrada_texto = f.read()
        print(entrada_texto)
except FileNotFoundError:
    print("⚠️ ERRO: entrada1.txt não existe!")
    exit()
print("===== FIM DO ARQUIVO =====\n")

# =========================
# PASSAR O ARQUIVO PARA O MODELO
# =========================

print("⏳ Rodando inferência...")

start = time.perf_counter()
resultado = inferir(entrada_texto)
tempo_execucao = time.perf_counter() - start


# =========================
# SALVAR RESULTADO
# =========================

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"TEMPO_DE_EXECUCAO={tempo_execucao}\n\n")
    f.write(resultado)

print(f"\n🟢 Resultado salvo em:\n{output_path}")
