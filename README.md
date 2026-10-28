# Projet IA locale : de Ollama à l'agent d'entreprise

Assistant interne fictif pour l'entreprise **NovaTech**, 100 % local grâce à [Ollama](https://ollama.com) :

1. **LLM local** avec Ollama
2. **RAG** (Retrieval-Augmented Generation) sur des documents internes
3. **Agent connecté** qui choisit ses outils : RAG, calculatrice, date, météo (API web)

## Architecture

```
Question ─► Agent (llama3.2 + tool calling)
               ├─► search_company_docs ─► RAG : embeddings (nomic-embed-text) + similarité cosinus
               ├─► calculator
               ├─► get_datetime
               └─► get_weather (wttr.in)
            ◄── Réponse sourcée en français
```

## Installation

### 1. Installer Ollama
- Linux : `curl -fsSL https://ollama.com/install.sh | sh`
- macOS / Windows : installateur sur https://ollama.com/download

### 2. Télécharger les modèles
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Installer le projet
```bash
git clone <URL_DU_DEPOT>
cd projet-ia-ollama
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

```bash
python rag.py ingest   # indexe les documents de docs/
python rag.py          # assistant RAG simple
python agent.py        # agent avec outils
```

### Exemples de questions
- RAG : « Combien de jours de RTT ai-je ? » / « Quel est le plafond d'un repas pro ? »
- Agent : « Il fait quel temps à Lyon ? » / « J'ai 25 jours de congés, combien de semaines ça fait ? »
  (l'agent enchaîne alors RAG puis calculatrice)

## Choix techniques
- **Pas de base vectorielle externe** : un index JSON + NumPy suffit pour un petit corpus et reste lisible.
- **Découpage** en passages de 800 caractères avec 150 de chevauchement pour ne pas couper les idées.
- **Prompt anti-hallucination** : le RAG répond uniquement avec le contexte, sinon il dit ne pas savoir.
- **Calculatrice sécurisée** : parsing AST, pas de `eval()`.

## Limites et pistes d'amélioration
- Index recalculé entièrement à chaque ingestion (pas d'incrémental).
- Ajouter PDF/DOCX, une interface Streamlit, ou une base vectorielle (Chroma, pgvector).
- Évaluer la qualité du RAG avec un jeu de questions de référence.
- Ajouter l'accès à un vrai outil métier (mail, calendrier, ticketing) via MCP.

## Démonstration
Lien vidéo / captures : _à compléter_
