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

miesiace = {
    "styczeń": 1,
    "luty": 2,
    "marzec": 3,
    "kwiecień": 4,
    "maj": 5,
    "czerwiec": 6,
    "lipiec": 7,
    "sierpień": 8,
    "wrzesień": 9,
    "październik": 10,
    "listopad": 11,
    "grudzień": 12
}

dlugosci_wyjazdu = {
    "jednodniowy": "duration_day_trip",
    "weekend": "duration_weekend",
    "krótki": "duration_short_trip",
    "tydzień": "duration_one_week",
    "długi": "duration_long_trip"
}

def score_temperature(actual_temp, preferred_temp):

    roznica = abs(actual_temp - preferred_temp)

    score = 1 - (roznica / 30)

    return max(score, 0)


def score_budget(destination_budget, user_budget):

    if destination_budget == user_budget:
        return 1.0

    if abs(destination_budget - user_budget) == 1:
        return 0.5

    return 0.0


def score_feature(destination_value, preferred_value):

    roznica = abs(destination_value - preferred_value)

    score = 1 - (roznica / 4)

    return max(score, 0)

def calculate_classic_score(
    dane,
    miesiac,
    preferowana_temperatura,
    budzet,
    dlugosc_wyjazdu,
    preferencje
):

    dane = dane.copy()

    numer_miesiaca = miesiace[miesiac.lower()]

    kolumna_temperatury = f"temp_{numer_miesiaca}"

    kolumna_dlugosci = dlugosci_wyjazdu[
        dlugosc_wyjazdu.lower()
    ]

    classic_scores = []

    for _, row in dane.iterrows():

        feature_scores = []

        for cecha, preferowana_wartosc in preferencje.items():

            score = score_feature(
                row[cecha],
                preferowana_wartosc
            )

            feature_scores.append(score)

        score_cech = sum(feature_scores) / len(feature_scores)

        score_temp = score_temperature(
            row[kolumna_temperatury],
            preferowana_temperatura
        )

        score_budzet = score_budget(
            row["budget_level_num"],
            budzet
        )

        score_dlugosc = row[kolumna_dlugosci]

        classic_score = (
            score_cech * 0.55
            + score_temp * 0.20
            + score_budzet * 0.15
            + score_dlugosc * 0.10
        )

        classic_scores.append(classic_score)

    dane["classic_score"] = classic_scores

    return dane

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

def hybrid_recommendation(
    dane,
    user_text,
    miesiac,
    preferowana_temperatura,
    budzet,
    dlugosc_wyjazdu,
    preferencje,
    classic_weight=0.7,
    nlp_weight=0.3,
    top_n=10
):

    results = calculate_classic_score(
        dane=dane,
        miesiac=miesiac,
        preferowana_temperatura=preferowana_temperatura,
        budzet=budzet,
        dlugosc_wyjazdu=dlugosc_wyjazdu,
        preferencje=preferencje
    )

    user_embedding = model.encode(
        [user_text]
    )

    similarities = cosine_similarity(
        user_embedding,
        destination_embeddings
    )[0]

    results["nlp_score"] = similarities

    min_nlp = results["nlp_score"].min()
    max_nlp = results["nlp_score"].max()

    results["nlp_score_normalized"] = (
        results["nlp_score"] - min_nlp
    ) / (
        max_nlp - min_nlp
    )

    results["hybrid_score"] = (
        results["classic_score"] * classic_weight
        + results["nlp_score_normalized"] * nlp_weight
    )


    results["classic_match_percent"] = (
        results["classic_score"] * 100
    ).round(1)

    results["nlp_match_percent"] = (
        results["nlp_score"] * 100
    ).round(1)

    results["hybrid_match_percent"] = (
        results["hybrid_score"] * 100
    ).round(1)


    results = results.sort_values(
        by="hybrid_score",
        ascending=False
    ).head(top_n)

    return results

preferencje_uzytkownika = {
    "culture": 4,
    "adventure": 3,
    "nature": 4,
    "beaches": 5,
    "nightlife": 3,
    "cuisine": 4,
    "wellness": 3,
    "urban": 2,
    "seclusion": 4
}

user_text = """
Chcę pojechać we wrześniu w ciepłe miejsce nad morzem.
Zależy mi na pięknych plażach, naturze i dobrej kuchni.
Lubię trochę zwiedzać, ale nie zależy mi na intensywnym
życiu nocnym. Wolę spokojniejszy wyjazd i odpoczynek.
"""

wybrany_miesiac = "wrzesień"

preferowana_temperatura = 26

budzet = 2

dlugosc_wyjazdu = "tydzień"


results = hybrid_recommendation(
    dane=df,
    user_text=user_text,
    miesiac=wybrany_miesiac,
    preferowana_temperatura=preferowana_temperatura,
    budzet=budzet,
    dlugosc_wyjazdu=dlugosc_wyjazdu,
    preferencje=preferencje_uzytkownika,
    classic_weight=0.7,
    nlp_weight=0.3,
    top_n=10
)


print("\nHybrydowe rekomendacje kierunków:\n")

print(
    results[
        [
            "city",
            "country",
            "cluster_name",
            "budget_level",
            "classic_match_percent",
            "nlp_match_percent",
            "hybrid_match_percent"
        ]
    ].to_string(
        index=False,
        formatters={
            "classic_match_percent": lambda x: f"{x:.1f}%",
            "nlp_match_percent": lambda x: f"{x:.1f}%",
            "hybrid_match_percent": lambda x: f"{x:.1f}%"
        }
    )
)