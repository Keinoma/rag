## RAG local (Qwen2.5-1.5B + Chroma)

Prototype de RAG qui répond à des questions sur un petit corpus de résumés de films,
avec un LLM local et un jeu de test pour évaluer le pipeline.

## Architecture du projet

rag/
├── rag.py (python rag.py index, puis : python rag.py chat OU python eval.py)  
├── eval.py (3 modes : RAG / sans RAG / oracle)
├── .gitignore
├── historique*des_commits.txt  
├── documents/ (10 fichiers .txt, un par film)
│ ├── 01-03 Pirates des Caraïbes (réels)
│ ├── 04-06 Kung Fu Panda (réels)
│ ├── 07-08 Films fictifs (aucune connaissance préalable possible)
│ └── 09-10 Distracteurs (Mulan, Sinbad)
├── documents non présentés/  
├── chroma_db/ (généré par `rag.py index`)
└── test/
├── questions.jsonl
└── runs/ (généré par `eval.py`)
└── <date>*<tag>/
├── results.jsonl # Détail complet : chunks, rangs, réponses, temps
├── review.csv # Relecture manuelle (colonnes verdict\_\* à remplir)
└── summary.json # Configuration du run + métriques agrégées

## Paramètres principaux (`rag.py`)

| Paramètre        | Valeur                     | Rôle                             |
| ---------------- | -------------------------- | -------------------------------- |
| `CHUNK_CHARS`    | 750                        | Taille d'un morceau              |
| `OVERLAP_CHARS`  | 300                        | Chevauchement entre morceaux     |
| `TOP_K`          | 5                          | Nombre d'extraits envoyés au LLM |
| `EMBED_MODEL`    | BAAI/bge-m3                | Modèle d'embeddings              |
| `LLM_NAME`       | Qwen/Qwen2.5-1.5B-Instruct | Modèle de génération (4 bits)    |
| `MAX_NEW_TOKENS` | 150                        | Longueur maximale de la réponse  |

## Les commits importants récents :

1: tout fonctionne +/-
Mieux de tout mettre 4 résumés dans un fichier plutôt que 1fhcihier/résumé

2:
Réduction de CHUNK_CHARS et TOP_K.
Surtout, ajout dans création de contexte d'un truc qui empêche de couper une phrase en 2. Attendre fin phrase.

3:
Production de 10 textes de réf
Production d'un lot de 50 quest/rep format .jsonl
