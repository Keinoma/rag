"""
Prototype RAG minimal.

    python rag.py index    -> lit le dossier documents/ et construit l'index
    python rag.py chat     -> discussion avec tes documents
"""
import json
import sys
from pathlib import Path

import chromadb
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TextStreamer)

# ============================ CONFIGURATION ============================
BASE = Path(__file__).parent
DOCS_DIR = BASE / "documents"      # tes PDF / txt / md
DB_DIR = BASE / "chroma_db"        # l'index sera stocké ici

EMBED_MODEL = "BAAI/bge-m3"        # embeddings multilingues (bon en français)
CHUNK_CHARS = 2000                 # ~500 tokens
OVERLAP_CHARS = 300                # chevauchement entre morceaux
TOP_K = 5                          # nombre d'extraits envoyés au LLM

# LLM : None = on lit le nom du modèle de base dans adapter_config.json.
# Sinon mets un nom ou un chemin, ex. "Qwen/Qwen2.5-3B-Instruct"
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
USE_ADAPTER = False                # True = applique ton adapter LoRA par-dessus
ADAPTER_DIR = Path(r"C:\Projets\finetune\modele_finetuned_lora")
LOAD_4BIT = False                   # 4 bits : économise la VRAM
MAX_NEW_TOKENS = 512
HISTORY_TURNS = 3                  # nombre d'échanges gardés en mémoire

SYSTEM_PROMPT = (
    "Tu es un assistant pédagogique. Réponds en français, uniquement à partir "
    "des extraits fournis. Cite tes sources avec [1], [2], etc. Si "
    "l'information n'est pas dans les extraits, dis-le clairement au lieu "
    "d'inventer."
)


# ============================ INDEXATION ============================
def extract_pages(path):
    """Renvoie une liste de (numéro_de_page, texte)."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def chunk(text, size=CHUNK_CHARS, overlap=OVERLAP_CHARS):
    """Découpe en morceaux d'environ `size` caractères, par lignes entières,
    avec un chevauchement de `overlap` caractères."""
    units = []
    for line in text.split("\n"):
        line = line.strip()
        while len(line) > size:  # ligne géante : découpe brute
            units.append(line[:size])
            line = line[size:]
        if line:
            units.append(line)

    chunks, current = [], []
    for u in units:
        if current and sum(len(x) + 1 for x in current) + len(u) > size:
            chunks.append("\n".join(current))
            tail, n = [], 0  # on garde la fin du morceau précédent
            for x in reversed(current):
                if n + len(x) > overlap:
                    break
                tail.insert(0, x)
                n += len(x) + 1
            current = tail
        current.append(u)
    if current:
        chunks.append("\n".join(current))
    return chunks


def build_index():
    files = sorted(p for p in DOCS_DIR.glob("**/*")
                   if p.suffix.lower() in {".pdf", ".txt", ".md"})
    if not files:
        sys.exit(f"Aucun document trouvé dans {DOCS_DIR}")

    ids, texts, metas = [], [], []
    for f in files:
        for page, text in extract_pages(f):
            for i, c in enumerate(chunk(text)):
                ids.append(f"{f.name}-p{page}-c{i}")
                texts.append(c)
                metas.append({"source": f.name, "page": page})
    if not texts:
        sys.exit("Aucun texte extrait (PDF scannés ? il faudrait de l'OCR).")
    print(f"{len(files)} document(s), {len(texts)} morceaux. Calcul des embeddings...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    embedder.max_seq_length = 1024
    vectors = embedder.encode(texts, batch_size=16, normalize_embeddings=True,
                              show_progress_bar=True)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection("docs")  # on repart de zéro à chaque indexation
    except Exception:
        pass
    col = client.create_collection("docs")
    for i in range(0, len(ids), 500):
        col.add(ids=ids[i:i + 500], documents=texts[i:i + 500],
                embeddings=vectors[i:i + 500].tolist(), metadatas=metas[i:i + 500])
    print(f"Index construit : {len(ids)} morceaux dans {DB_DIR}")


# ============================ LLM ============================
def load_llm():
    name = LLM_NAME
    if name is None:
        cfg = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())
        name = cfg["base_model_name_or_path"]
    print(f"Modèle de base : {name}")
    if "instruct" not in str(name).lower():
        print("ATTENTION : ce n'est pas une version 'Instruct', "
              "le suivi des consignes sera moins bon.")

    tok = AutoTokenizer.from_pretrained(name)
    kwargs = dict(device_map={"": 0}, dtype=torch.bfloat16)
    if LOAD_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if USE_ADAPTER:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
        print("Adapter LoRA appliqué.")
    model.eval()
    return tok, model


def generate(tok, model, messages, max_new_tokens=MAX_NEW_TOKENS, stream=False):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True) if stream else None
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None, streamer=streamer)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ============================ CONVERSATION ============================
def rewrite_question(tok, model, history, question):
    """Rend la question compréhensible seule (« et pour le deuxième ? »)."""
    if not history:
        return question
    convo = "\n".join(f"Utilisateur : {q}\nAssistant : {a[:300]}"
                      for q, a in history[-HISTORY_TURNS:])
    msgs = [
        {"role": "system", "content": "Réécris la dernière question de l'utilisateur "
         "pour qu'elle se comprenne seule, en t'aidant de la conversation. "
         "Réponds uniquement par la question réécrite."},
        {"role": "user", "content": f"Conversation :\n{convo}\n\nDernière question : {question}"},
    ]
    return generate(tok, model, msgs, max_new_tokens=80) or question


def retrieve(col, embedder, query):
    vec = embedder.encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=vec, n_results=TOP_K)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def chat():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        col = client.get_collection("docs")
    except Exception:
        sys.exit("Aucun index trouvé. Lance d'abord : python rag.py index")

    print("Chargement des modèles...")
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")  # GPU gardé pour le LLM
    tok, model = load_llm()

    history = []  # liste de (question, réponse)
    print("\nPrêt. Pose tes questions (ligne vide ou 'quit' pour quitter).\n")
    while True:
        question = input("Toi > ").strip()
        if question.lower() in {"", "quit", "exit"}:
            break

        search_query = rewrite_question(tok, model, history, question)
        if search_query != question:
            print(f"  (recherche : {search_query})")
        hits = retrieve(col, embedder, search_query)

        context = "\n\n".join(
            f"[{i}] ({m['source']}, p.{m['page']})\n{doc}"
            for i, (doc, m) in enumerate(hits, 1))
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for q, a in history[-HISTORY_TURNS:]:
            messages += [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]
        messages.append({"role": "user",
                         "content": f"Extraits :\n{context}\n\nQuestion : {question}"})

        print("\nAssistant > ", end="", flush=True)
        answer = generate(tok, model, messages, stream=True)
        history.append((question, answer))

        print("\nSources :")
        for i, (_, m) in enumerate(hits, 1):
            print(f"  [{i}] {m['source']}, p.{m['page']}")
        print()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "index":
        build_index()
    elif cmd == "chat":
        chat()
    else:
        print("Usage : python rag.py index | python rag.py chat")