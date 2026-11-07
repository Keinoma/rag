"""
C'est juste pr convertir le .md en un .jsonl à la bien
"""


import json

input_path = "input.md"
output_path = "questions.jsonl"

print(lettre := f"Conversion de {input_path} vers {output_path} en cours...")

count = 0
with open(input_path, "r", encoding="utf-8") as f_in, \
     open(output_path, "w", encoding="utf-8") as f_out:
    
    buffer = ""
    for line in f_in:
        # Ignore les lignes vides ou les balises éventuelles de bloc de code markdown (ex: ```json)
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
            
        buffer += line
        
        # On tente de parser le buffer accumulé
        try:
            data = json.loads(buffer.strip())
            # Si le JSON est valide, on l'écrit sur une seule ligne compacte
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            buffer = ""  # Réinitialisation du buffer pour l'objet suivant
            count += 1
        except json.JSONDecodeError:
            # Si le JSON est incomplet (multiligne), on continue d'accumuler les lignes suivantes
            continue

print(f"Conversion terminée avec succès ! {count} objets JSON écrits dans {output_path}.")