"""Agent connecté : le LLM choisit lui-même ses outils (RAG, heure, calcul, météo web)."""
import ast
import json
import operator
import urllib.request
from datetime import datetime

import ollama

from rag import LLM_MODEL, retrieve


def search_company_docs(query: str) -> str:
    """Cherche dans les documents internes de l'entreprise (congés, support, notes de frais).

    Args:
        query: la question ou les mots-clés à rechercher.
    """
    return "\n---\n".join(f"[{p['source']}] {p['text']}" for p in retrieve(query, k=3))


def get_datetime() -> str:
    """Donne la date et l'heure actuelles."""
    return datetime.now().strftime("%A %d %B %Y, %H:%M")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Expression non autorisée")


def calculator(expression: str) -> str:
    """Calcule une expression arithmétique, par exemple '25 * 12 + 3'.

    Args:
        expression: l'expression mathématique à évaluer.
    """
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Erreur : {e}"


def get_weather(city: str) -> str:
    """Donne la météo actuelle d'une ville via wttr.in (accès internet).

    Args:
        city: nom de la ville, par exemple 'Lyon'.
    """
    try:
        url = f"https://wttr.in/{city}?format=j1"
        with urllib.request.urlopen(url, timeout=10) as r:
            cur = json.load(r)["current_condition"][0]
        return f"{city} : {cur['temp_C']}°C, ressenti {cur['FeelsLikeC']}°C, humidité {cur['humidity']}%"
    except Exception as e:
        return f"Météo indisponible : {e}"


TOOLS = {f.__name__: f for f in (search_company_docs, get_datetime, calculator, get_weather)}


def run(question: str, max_steps: int = 5) -> str:
    messages = [
        {"role": "system", "content": "Tu es l'assistant de l'entreprise NovaTech. "
         "Réponds en français. Utilise les outils quand c'est utile."},
        {"role": "user", "content": question},
    ]
    for _ in range(max_steps):
        res = ollama.chat(model=LLM_MODEL, messages=messages, tools=list(TOOLS.values()))
        messages.append(res["message"])
        calls = res["message"].get("tool_calls")
        if not calls:
            return res["message"]["content"]
        for call in calls:
            name, args = call["function"]["name"], call["function"]["arguments"]
            print(f"  [outil] {name}({args})")
            result = TOOLS[name](**args) if name in TOOLS else "Outil inconnu"
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})
    return "Nombre maximal d'étapes atteint."


if __name__ == "__main__":
    print("Agent NovaTech (Ctrl+C pour quitter)")
    while True:
        print("\n" + run(input("\nVous > ")))
