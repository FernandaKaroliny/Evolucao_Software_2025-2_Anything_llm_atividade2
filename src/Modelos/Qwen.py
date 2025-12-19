# -*- coding: utf-8 -*-

import os
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

print("GPU disponível?", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("GPU não disponível. Ative uma GPU ou use CUDA.")


# =========================
# CONFIGURAÇÃO DO MODELO
# =========================

model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"

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


def inferir(texto):
    instruction = '''Foram dados acima: Trecho do README sobre colaboração e estrutura
                     de branches. Baseando-se nesses, defina qual é a estratégia de branch que está
                     sendo utilizada.'''

    messages = [
        {"role": "system", "content": "Você é especialista em engenharia de software."},
        {"role": "user", "content": texto},
        {"role": "user", "content": instruction}
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
# CONFIGURAR PASTAS (entradas/ e resultados/)
# =========================

base_dir = os.path.dirname(os.path.abspath(__file__))

entradas_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "entradas"))
respostas_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "resultados"))

os.makedirs(entradas_dir, exist_ok=True)
os.makedirs(respostas_dir, exist_ok=True)

output_path = os.path.join(respostas_dir, "Qwen.txt")
input_path = os.path.join(entradas_dir, "entrada5.txt")

# Ler e imprimir o conteúdo existente do arquivo
print("\n===== CONTEÚDO DE entrada5.txt =====")
try:
    with open(input_path, "r", encoding="utf-8") as f:
        entrada_texto = f.read()
        print(entrada_texto)
except FileNotFoundError:
    print("ERRO: entrada5.txt não existe!")
    exit()
print("===== FIM DO ARQUIVO =====\n")

# =========================
# PASSAR O ARQUIVO PARA O MODELO
# =========================

print("Rodando inferência...")

start = time.perf_counter()
resultado = inferir(entrada_texto)
tempo_execucao = time.perf_counter() - start


# =========================
# SALVAR RESULTADO
# =========================

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"TEMPO_DE_EXECUCAO={tempo_execucao}\n\n")
    f.write(resultado)

print(f"\nResultado salvo em:\n{output_path}")
