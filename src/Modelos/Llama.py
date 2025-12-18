import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os

prompts_testados = 0
#estrutura = lt.listar_estrutura("https://github.com/Mintplex-Labs/anything-llm") quando quiser listar a estrutura do repositório
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARQUIVO_SAIDA_TXT = os.path.join(BASE_DIR, "..", "..", "resultados", "Llama1")



entradas_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "entradas"))
os.makedirs(entradas_dir, exist_ok=True)




# Exibe uso de memória
def mostrar_uso_memoria():
    processo = psutil.Process()
    mem_info = processo.memory_info().rss / (1024 ** 2)
    print(f" Memória RAM usada: {mem_info:.2f} MB")
    if device == "cuda":
        print(f"Memória GPU usada: {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB / "
              f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 2):.2f} MB totais")
        
def executarModelo(input_text):
   global prompts_testados
   prompts_testados += 1
   texto_avaliado = input_text
   inputs = tokenizer(texto_avaliado, return_tensors="pt").to(model.device)
  
   tempo_execucao = time.perf_counter()
   #outputs = model.generate(**inputs, max_new_tokens=700, temperature=0.15, top_p=0.9)
   outputs = model.generate(
    **inputs,
    max_new_tokens=700,
    temperature=0.15,
    do_sample=False,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id

   )
   tempo_execucao = time.perf_counter() - tempo_execucao

   # Decodifica o texto completo
   decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
   print("\n_____________________________________________")
   print(f"Prompt testado nº: {prompts_testados}")
   print(f"\nTempo de execução do modelo: {tempo_execucao:.2f} segundos")
   mostrar_uso_memoria()
   print("\n" + decoded + "\n_____________________________________________  " +  f"\nTempo de execução do modelo: {tempo_execucao:.2f} segundos")
   return decoded  # Extrai apenas a resposta gerada


def prompt1():
   ARQUIVO_ENTRADA_TXT = os.path.join(entradas_dir, "entrada2.txt")
   
   print("arquivo de entrada:", ARQUIVO_ENTRADA_TXT)
   with open(ARQUIVO_ENTRADA_TXT, "r", encoding="utf-8") as f:
      global prompt
      arquivo = f.read()

      prompt = f"""

      {arquivo}
      Com base nas dependências, scripts e estrutura, descreva:
      1. Quais tecnologias principais o projeto utiliza (ex: Node.js, React, Express, Prisma, etc.).
      2. Qual o tipo de arquitetura que o projeto segue (ex: Monolítico, MVC, Microservices, Event-Driven, Client-Server, etc.).
      3. Quais diretórios ou pacotes indicam separação de responsabilidades (ex: 'server', 'frontend', 'collector').
      4. Como ocorre o fluxo de execução em desenvolvimento (baseando-se nos scripts `dev:*` e `setup`).
      5. Se o projeto parece voltado para execução local, em nuvem, ou ambos.
      6. Quais boas práticas ou convenções de arquitetura são evidentes (ex: uso de Prisma para ORM, divisão entre back e front, uso de Docker, etc.).
      7. Gere um pequeno resumo final explicando o propósito geral do projeto e seu padrão arquitetural principal.

      Responda em português de forma clara OBJETIVA, não repetitiva além de Avaliar a confiabilidade da sua resposta.
      """
      
      print(prompt)
      
      arquivo = ARQUIVO_SAIDA_TXT + "_analise_arquitetura.txt"



      resposta = executarModelo(prompt)
      with open(arquivo, "w", encoding="utf-8") as f: 
         f.write(resposta)
      
      # Mostra uso de memória após geração

def prompt2():
   ARQUIVO_ENTRADA_TXT = os.path.join(entradas_dir, "entrada5.txt")
   
   print("arquivo de entrada:", ARQUIVO_ENTRADA_TXT)
   with open(ARQUIVO_ENTRADA_TXT, "r", encoding="utf-8") as f:
      global prompt
      arquivo = f.read()

      prompt = f"""
      Start of input file:
      {arquivo}
      End of input file.

      You are a specialist in Software Engineering and Open Source Project Governance.

      TASK:
      Analyze the Mintplex-Labs/anything-llm project based exclusively on the content of the input file provided.

      STRICT CONSTRAINTS:
      - Use ONLY the information explicitly present in the input file.
      - Do NOT use prior knowledge or assumptions about the project.
      - If some information cannot be determined from the input, explicitly state:
      "Not observable from the provided input."
      - Do NOT repeat or paraphrase the prompt.
      - Do NOT introduce new questions or topics.
      - Answer ONLY the questions listed below.

      QUESTIONS:
      1. Which Release Strategy is used by the project?
         (Rapid Releases, Release Train, or LTS + Current)

      2. Which Workflow Model (Branching Model) is adopted?
         (Gitflow, GitHub Flow, or another model)

      OUTPUT FORMAT (MANDATORY):
      1. Release Strategy:
         Answer:
         Evidence from the input:

      2. Workflow Model:
         Answer:
         Evidence from the input:

      """

      prompt_traduzido = f"""
      Start of input file:
         {arquivo}
         End of input file.

         Você é um especialista em Engenharia de Software e Governança de Projetos Open Source.

         TAREFA:
         Analise o projeto Mintplex-Labs/anything-llm exclusivamente com base no conteúdo do arquivo de entrada fornecido.

         RESTRIÇÕES ESTRITAS:
         - Utilize APENAS as informações explicitamente presentes no arquivo de entrada.
         - NÃO utilize conhecimento prévio ou suposições sobre o projeto.
         - Se alguma informação não puder ser determinada a partir da entrada, declare explicitamente:
         "Não observável a partir da entrada fornecida."
         - NÃO repita nem parafraseie o prompt.
         - NÃO introduza novas perguntas ou tópicos.
         - Responda APENAS às perguntas listadas abaixo.

         PERGUNTAS:
         1. Qual Estratégia de Releases é utilizada pelo projeto?
            (Rapid Releases, Release Train ou LTS + Current)

         2. Qual Modelo de Fluxo de Trabalho (Branching Model) é adotado?
            (Gitflow, GitHub Flow ou outro modelo)

         FORMATO DE SAÍDA (OBRIGATÓRIO):
         1. Estratégia de Releases:
            Resposta:
            Evidência extraída da entrada:

         2. Modelo de Fluxo de Trabalho:
            Resposta:
            Evidência extraída da entrada:

      """

      print(prompt)
      arquivo = ARQUIVO_SAIDA_TXT + "analise.txt"
      resposta = executarModelo(prompt)
      with open(arquivo, "w", encoding="utf-8") as f: 
            f.write(resposta)



model_path = "meta-llama/Llama-3.2-1B-Instruct"
# Utilizei um modelo baixado localmente, altere caso o modelo 
# precise ser baixado do Huggingface, ou caso o modelo
# é preciso solicitar a autorização da meta para uso

print("Carregando modelo", model_path)
device = "cpu"
# Detecta GPU disponível
if torch.cuda.is_available():
   device = "cuda"
   print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
else:
   device = "cpu"
   print("Nenhuma GPU detectada, rodando na CPU")
#device = "cpu"

# Carrega modelo e tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)


mostrar_uso_memoria()

while (True):
   print("Escolha o prompt que deseja utilizar:")
   print("1 - Análise de arquitetura de projeto de software")
   print("2 - Análise de estratégia de releases e fluxo de trabalho")
   op = input("Digite o número do prompt (ou '0' para encerrar): ")
   if op == "0":
      break
   elif op == "1":
      prompt = prompt1()
   elif op == "2":
      prompt = prompt2()




