## Architecture du projet
rag/
├── rag.py                          
├── eval.py  (3 modes : RAG / sans RAG / oracle)
├── .gitignore
├── historique_des_commits.txt    
│
├── documents/  (10 fichiers .txt, un par film)
│   ├── 01-03  Pirates des Caraïbes (réels)
│   ├── 04-06  Kung Fu Panda (réels)
│   ├── 07-08  Films fictifs (aucune connaissance préalable possible)
│   └── 09-10  Distracteurs (Mulan, Sinbad)
│
├── documents non présentés/      
│
├── chroma_db/  (généré par `rag.py index`)
│
└── test/
    ├── questions.jsonl
    └── runs/  (généré par `eval.py`)
        └── <date>_<tag>/
            ├── results.jsonl       # Détail complet : chunks, rangs, réponses, temps
            ├── review.csv          # Relecture manuelle (colonnes verdict_* à remplir)
            └── summary.json        # Configuration du run + métriques agrégées





