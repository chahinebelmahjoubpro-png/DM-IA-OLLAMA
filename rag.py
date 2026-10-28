"""Pipeline RAG local avec Ollama (embeddings + génération), sans base externe."""
import json
import sys
from pathlib import Path

import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
DOCS_DIR = Path("docs")
INDEX_FILE = Path("index.json")


def chunk(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Découpe un texte en morceaux qui se chevauchent légèrement."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def build_index() -> None:
    """Lit les documents, les découpe, calcule les embeddings et sauvegarde l'index."""
    items = []
    for path in sorted(DOCS_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        for i, part in enumerate(chunk(path.read_text(encoding="utf-8"))):
            items.append({"source": path.name, "chunk": i, "text": part})
    if not items:
        sys.exit("Aucun document trouvé dans docs/")
    vectors = embed([it["text"] for it in items])
    for it, vec in zip(items, vectors):
        it["embedding"] = vec
    INDEX_FILE.write_text(json.dumps(items), encoding="utf-8")
    print(f"Index construit : {len(items)} passages.")


def retrieve(query: str, k: int = 4) -> list[dict]:
    """Retourne les k passages les plus proches de la question (similarité cosinus)."""
    if not INDEX_FILE.exists():
        build_index()
    items = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    matrix = np.array([it["embedding"] for it in items])
    q = np.array(embed([query])[0])
    scores = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q))
    best = np.argsort(scores)[::-1][:k]
    return [{**items[i], "score": float(scores[i])} for i in best]


def answer(question: str) -> tuple[str, list[dict]]:
    """Répond à la question en s'appuyant uniquement sur les passages retrouvés."""
    passages = retrieve(question)
    context = "\n\n".join(f"[{p['source']}]\n{p['text']}" for p in passages)
    prompt = (
        "Réponds en français en t'appuyant UNIQUEMENT sur le contexte ci-dessous. "
        "Si la réponse n'y est pas, dis que tu ne sais pas.\n\n"
        f"Contexte :\n{context}\n\nQuestion : {question}"
    )
    res = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return res["message"]["content"], passages


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        build_index()
    else:
        print("Assistant RAG (Ctrl+C pour quitter)")
        while True:
            q = input("\nQuestion > ")
            text, sources = answer(q)
            print("\n" + text)
            print("\nSources :", ", ".join(sorted({s["source"] for s in sources})))
