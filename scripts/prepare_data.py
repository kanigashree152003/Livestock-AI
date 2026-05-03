import json
import os

os.makedirs("rag", exist_ok=True)

# -------------------------------
# LOAD ORIGINAL DATA
# -------------------------------
with open("lesion_knowledge.json", "r") as f:
    data = json.load(f)

processed_docs = []

# -------------------------------
# PROCESS ORIGINAL 3 DOCS
# -------------------------------
for item in data:
    disease = item["metadata"]["disease"]
    species = ", ".join(item["metadata"]["species"])
    severity = item["metadata"]["severity"]
    contagious = item["metadata"]["contagious"]

    enriched_text = f"""
Disease: {disease}

Species: {species}

Description:
{item["text"]}

Common symptoms:
itching, lesions, swelling, skin patches, wounds, mouth sores

Possible body parts affected:
eye, face, mouth, skin, neck, legs

Severity: {severity}
Contagious: {contagious}

Keywords:
goat disease, sheep disease, infection, skin disease, lesion, treatment, prevention
"""

    processed_docs.append(enriched_text.strip())

# -------------------------------
# LOAD EXTRA 50 DOCS
# -------------------------------
with open("extra_docs.json", "r") as f:
    extra_docs = json.load(f)

# 🔥 IMPORTANT FIX
for doc in extra_docs:
    processed_docs.append(doc["text"])

# -------------------------------
# SAVE FINAL FILE
# -------------------------------
with open("rag/processed_documents.json", "w") as f:
    json.dump(processed_docs, f, indent=2)

print("✅ Data enriched and saved successfully")