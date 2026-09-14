import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "Worldwide Travel Cities Dataset (Ratings and Climate).csv"

df = pd.read_csv(DATA_PATH)

print("Rozmiar datasetu:")
print(df.shape)

print("\nPierwsze 5 wierszy:")
print(df.head())

print("\nNazwy kolumn:")
print(df.columns.tolist())

print("\nTypy danych:")
print(df.dtypes)

print("\nBraki danych:")
print(df.isnull().sum())

print("\nLiczba duplikatów:")
print(df.duplicated().sum())

print("\nUnikalne regiony:")
print(df["region"].value_counts())

print("\nPoziomy budżetu:")
print(df["budget_level"].value_counts())

print("\nPrzykładowe wartości ideal_durations:")
print(df["ideal_durations"].head(20))

print("\nPrzykładowe wartości avg_temp_monthly:")
print(df["avg_temp_monthly"].head(10))

print("\nZakres ocen:")
rating_columns = [
    "culture",
    "adventure",
    "nature",
    "beaches",
    "nightlife",
    "cuisine",
    "wellness",
    "urban",
    "seclusion"
]

print(df[rating_columns].describe())

import json


budget_mapping = {
    "Budget": 1,
    "Mid-range": 2,
    "Luxury": 3
}

df["budget_level_num"] = df["budget_level"].map(budget_mapping)

print("\nBudget level po zamianie:")
print(df[["budget_level", "budget_level_num"]].head())


def extract_monthly_avg_temp(temp_json):
    temp_data = json.loads(temp_json)

    result = {}

    for month in range(1, 13):
        result[f"temp_{month}"] = temp_data[str(month)]["avg"]

    return pd.Series(result)


temperature_columns = df["avg_temp_monthly"].apply(extract_monthly_avg_temp)

df = pd.concat([df, temperature_columns], axis=1)

print("\nTemperatury miesięczne:")
print(
    df[
        [
            "city",
            "temp_1",
            "temp_2",
            "temp_3",
            "temp_4",
            "temp_5",
            "temp_6",
            "temp_7",
            "temp_8",
            "temp_9",
            "temp_10",
            "temp_11",
            "temp_12"
        ]
    ].head()
)

df["ideal_durations_list"] = df["ideal_durations"].apply(json.loads)

print("\nPrzykładowe idealne długości wyjazdu:")
print(df[["city", "ideal_durations_list"]].head())

duration_categories = [
    "Day trip",
    "Weekend",
    "Short trip",
    "One week",
    "Long trip"
]

for duration in duration_categories:
    column_name = (
        "duration_"
        + duration.lower().replace(" ", "_")
    )

    df[column_name] = df["ideal_durations_list"].apply(
        lambda durations: 1 if duration in durations else 0
    )


print("\nKolumny długości wyjazdu:")
print(
    df[
        [
            "city",
            "duration_day_trip",
            "duration_weekend",
            "duration_short_trip",
            "duration_one_week",
            "duration_long_trip"
        ]
    ].head()
)


PROCESSED_PATH = BASE_DIR / "data" / "processed" / "destinations_processed.csv"

df.to_csv(PROCESSED_PATH, index=False)

print("\nPrzetworzony dataset zapisano do:")
print(PROCESSED_PATH)

print("\nNowy rozmiar datasetu:")
print(df.shape)