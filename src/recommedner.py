import pandas as pd
from pathlib import Path

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

nazwy_cech = {
    "culture": "kultura",
    "adventure": "przygoda",
    "nature": "natura",
    "beaches": "plaże",
    "nightlife": "życie nocne",
    "cuisine": "kuchnia",
    "wellness": "wellness",
    "urban": "miejski charakter",
    "seclusion": "spokój i odosobnienie"
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


def create_explanation(
    row,
    preferencje,
    score_temp,
    score_budzet,
    score_dlugosc,
    kolumna_temperatury,
    preferowana_temperatura
):

    wyjasnienia = []

    rzeczywista_temp = row[kolumna_temperatury]

    roznica_temp = abs(
        rzeczywista_temp - preferowana_temperatura
    )

    if roznica_temp <= 2:
        wyjasnienia.append(
            f"bardzo dobre dopasowanie temperatury "
            f"({rzeczywista_temp:.1f}°C)"
        )

    elif roznica_temp <= 5:
        wyjasnienia.append(
            f"dobre dopasowanie temperatury "
            f"({rzeczywista_temp:.1f}°C)"
        )

    else:
        wyjasnienia.append(
            f"temperatura częściowo odbiega od preferencji "
            f"({rzeczywista_temp:.1f}°C)"
        )


    if score_budzet == 1:
        wyjasnienia.append(
            "poziom budżetu jest zgodny z preferencjami"
        )

    elif score_budzet == 0.5:
        wyjasnienia.append(
            "poziom budżetu jest zbliżony do preferencji"
        )

    else:
        wyjasnienia.append(
            "poziom budżetu różni się od preferencji"
        )


    if score_dlugosc == 1:
        wyjasnienia.append(
            "kierunek pasuje do wybranej długości wyjazdu"
        )

    else:
        wyjasnienia.append(
            "wybrana długość wyjazdu nie jest wskazana jako idealna"
        )

    dopasowania_cech = []

    for cecha, preferowana_wartosc in preferencje.items():

        wynik_cechy = score_feature(
            row[cecha],
            preferowana_wartosc
        )

        dopasowania_cech.append(
            (
                cecha,
                wynik_cechy,
                row[cecha]
            )
        )


    dopasowania_cech.sort(
        key=lambda x: x[1],
        reverse=True
    )


    najlepsze_cechy = dopasowania_cech[:3]

    opis_cech = []

    for cecha, wynik, wartosc in najlepsze_cechy:

        opis_cech.append(
            f"{nazwy_cech[cecha]} ({int(wartosc)}/5)"
        )


    wyjasnienia.append(
        "najlepiej dopasowane cechy: "
        + ", ".join(opis_cech)
    )


    return wyjasnienia

def recommend_destinations(
    dane,
    miesiac,
    preferowana_temperatura,
    budzet,
    dlugosc_wyjazdu,
    preferencje,
    top_n=10
):

    dane = dane.copy()

    numer_miesiaca = miesiace[miesiac.lower()]

    kolumna_temperatury = f"temp_{numer_miesiaca}"

    kolumna_dlugosci = dlugosci_wyjazdu[
        dlugosc_wyjazdu.lower()
    ]

    recommendation_scores = []
    feature_scores_list = []
    temperature_scores = []
    budget_scores = []
    duration_scores = []


    for _, row in dane.iterrows():

        feature_scores = []

        for cecha, preferowana_wartosc in preferencje.items():

            score = score_feature(
                row[cecha],
                preferowana_wartosc
            )

            feature_scores.append(score)


        score_cech = (
            sum(feature_scores)
            / len(feature_scores)
        )

        score_temp = score_temperature(
            row[kolumna_temperatury],
            preferowana_temperatura
        )

        score_budzet = score_budget(
            row["budget_level_num"],
            budzet
        )

        score_dlugosc = row[kolumna_dlugosci]

        final_score = (
            score_cech * 0.55
            + score_temp * 0.20
            + score_budzet * 0.15
            + score_dlugosc * 0.10
        )


        recommendation_scores.append(final_score)
        feature_scores_list.append(score_cech)
        temperature_scores.append(score_temp)
        budget_scores.append(score_budzet)
        duration_scores.append(score_dlugosc)


    dane["recommendation_score"] = recommendation_scores

    dane["feature_score"] = feature_scores_list
    dane["temperature_score"] = temperature_scores
    dane["budget_score"] = budget_scores
    dane["duration_score"] = duration_scores


    dane["match_percent"] = (
        dane["recommendation_score"] * 100
    ).round(1)


    wyniki = dane.sort_values(
        by="recommendation_score",
        ascending=False
    ).head(top_n).copy()

    wyjasnienia = []

    for _, row in wyniki.iterrows():

        explanation = create_explanation(
            row=row,
            preferencje=preferencje,
            score_temp=row["temperature_score"],
            score_budzet=row["budget_score"],
            score_dlugosc=row["duration_score"],
            kolumna_temperatury=kolumna_temperatury,
            preferowana_temperatura=preferowana_temperatura
        )

        wyjasnienia.append(explanation)


    wyniki["explanation"] = wyjasnienia

    return wyniki, kolumna_temperatury


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

wybrany_miesiac = "wrzesień"
preferowana_temperatura = 26
budzet = 2
dlugosc_wyjazdu = "tydzień"

wyniki, kolumna_temperatury = recommend_destinations(
    dane=df,
    miesiac=wybrany_miesiac,
    preferowana_temperatura=preferowana_temperatura,
    budzet=budzet,
    dlugosc_wyjazdu=dlugosc_wyjazdu,
    preferencje=preferencje_uzytkownika,
    top_n=10
)

print("\nNajlepiej dopasowane kierunki:\n")

print(
    wyniki[
        [
            "city",
            "country",
            "cluster_name",
            "budget_level",
            kolumna_temperatury,
            "match_percent"
        ]
    ].to_string(index=False)
)

print("\n")
print("=" * 70)
print("SZCZEGÓŁOWE WYJAŚNIENIE REKOMENDACJI")
print("=" * 70)


for numer, (_, row) in enumerate(
    wyniki.iterrows(),
    start=1
):

    print(
        f"\n{numer}. {row['city']}, "
        f"{row['country']} "
        f"— {row['match_percent']:.1f}% dopasowania"
    )

    print(
        f"   Typ kierunku: {row['cluster_name']}"
    )

    print(
        f"   Budżet: {row['budget_level']}"
    )

    print(
        f"   Temperatura: "
        f"{row[kolumna_temperatury]:.1f}°C"
    )

    print("   Dlaczego ten kierunek:")

    for powod in row["explanation"]:
        print(f"   - {powod}")