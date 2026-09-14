import pandas as pd
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "destinations_clustered.csv"
)

df = pd.read_csv(DATA_PATH)

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def create_destination_description(row):

    description = (
        f"{row['city']} in {row['country']}. "
        f"{row['short_description']} "
        f"Travel profile: "
        f"culture {row['culture']}/5, "
        f"adventure {row['adventure']}/5, "
        f"nature {row['nature']}/5, "
        f"beaches {row['beaches']}/5, "
        f"nightlife {row['nightlife']}/5, "
        f"cuisine {row['cuisine']}/5, "
        f"wellness {row['wellness']}/5, "
        f"urban character {row['urban']}/5, "
        f"seclusion {row['seclusion']}/5. "
        f"Budget level: {row['budget_level']}. "
        f"Destination type: {row['cluster_name']}."
    )

    return description

df["nlp_description"] = df.apply(
    create_destination_description,
    axis=1
)


print("Tworzenie embeddingów kierunków...")

destination_embeddings = model.encode(
    df["nlp_description"].tolist(),
    show_progress_bar=True
)

def recommend_from_text(user_text, top_n=10):

    user_embedding = model.encode(
        [user_text]
    )

    similarities = cosine_similarity(
        user_embedding,
        destination_embeddings
    )[0]

    results = df.copy()

    results["nlp_similarity"] = similarities

    results["nlp_match_percent"] = (
        results["nlp_similarity"] * 100
    ).round(1).astype(float)

    results = results.sort_values(
        by="nlp_similarity",
        ascending=False
    ).head(top_n)

    return results

user_text = """
Chcę pojechać we wrześniu w ciepłe miejsce nad morzem.
Zależy mi na pięknych plażach, naturze i dobrej kuchni.
Lubię trochę zwiedzać, ale nie zależy mi na intensywnym
życiu nocnym. Wolę spokojniejszy wyjazd i odpoczynek.
"""

results = recommend_from_text(
    user_text=user_text,
    top_n=10
)

print("\nRekomendacje na podstawie opisu tekstowego:\n")

print(
    results[
        [
            "city",
            "country",
            "cluster_name",
            "budget_level",
            "nlp_match_percent"
        ]
    ].to_string(
        index=False,
        formatters={
            "nlp_match_percent": lambda x: f"{x:.1f}%"
        }
    )
)