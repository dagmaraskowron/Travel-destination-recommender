import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "destinations_processed.csv"
)

IMAGES_DIR = BASE_DIR / "images"

PASTELOWY_ROZ = "#F6C1D1"
PASTELOWY_FIOLET = "#CDB4DB"

df = pd.read_csv(DATA_PATH)

print("Rozmiar zbioru danych:")
print(df.shape)

print("\nPierwsze wiersze:")
print(df.head())


liczba_regionow = df["region"].value_counts()

nazwy_regionow = {
    "europe": "Europa",
    "north_america": "Ameryka Północna",
    "asia": "Azja",
    "africa": "Afryka",
    "south_america": "Ameryka Południowa",
    "oceania": "Oceania",
    "middle_east": "Bliski Wschód"
}

liczba_regionow.index = (
    liczba_regionow.index.map(nazwy_regionow)
)

plt.figure(figsize=(10, 6))

liczba_regionow.plot(
    kind="bar",
    color=PASTELOWY_ROZ,
    edgecolor="white"
)

plt.title(
    "Liczba kierunków turystycznych według regionu"
)

plt.xlabel("Region")
plt.ylabel("Liczba kierunków")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "kierunki_wedlug_regionu_pastel.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

liczba_budzetow = (
    df["budget_level"]
    .value_counts()
)

nazwy_budzetow = {
    "Budget": "Budżetowy",
    "Mid-range": "Średni",
    "Luxury": "Luksusowy"
}

liczba_budzetow.index = (
    liczba_budzetow.index.map(nazwy_budzetow)
)

plt.figure(figsize=(8, 5))

liczba_budzetow.plot(
    kind="bar",
    color=PASTELOWY_ROZ,
    edgecolor="white"
)

plt.title(
    "Rozkład poziomów budżetu"
)

plt.xlabel("Poziom budżetu")
plt.ylabel("Liczba kierunków")

plt.xticks(rotation=0)

plt.grid(
    axis="y",
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "rozklad_budzetu_pastel.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

kolumny_ocen = [
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

srednie_oceny = (
    df[kolumny_ocen]
    .mean()
    .sort_values(
        ascending=False
    )
)

nazwy_cech = {
    "culture": "Kultura",
    "adventure": "Przygoda",
    "nature": "Natura",
    "beaches": "Plaże",
    "nightlife": "Życie nocne",
    "cuisine": "Kuchnia",
    "wellness": "Wellness",
    "urban": "Miejski charakter",
    "seclusion": "Spokój"
}

srednie_oceny.index = (
    srednie_oceny.index.map(
        nazwy_cech
    )
)

plt.figure(figsize=(10, 6))

srednie_oceny.plot(
    kind="bar",
    color=PASTELOWY_ROZ,
    edgecolor="white"
)

plt.title(
    "Średnie oceny cech kierunków turystycznych"
)

plt.xlabel("Cecha")
plt.ylabel("Średnia ocena")

plt.ylim(0, 5)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "srednie_oceny_cech_pastel.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

macierz_korelacji = (
    df[kolumny_ocen]
    .corr()
)

plt.figure(figsize=(9, 7))

obraz = plt.imshow(
    macierz_korelacji,
    aspect="auto",
    cmap="PuRd",
    vmin=-1,
    vmax=1
)

plt.colorbar(
    obraz,
    label="Współczynnik korelacji"
)

polskie_nazwy_cech = [
    nazwy_cech[kolumna]
    for kolumna in kolumny_ocen
]

plt.xticks(
    range(len(kolumny_ocen)),
    polskie_nazwy_cech,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(kolumny_ocen)),
    polskie_nazwy_cech
)

plt.title(
    "Korelacje między cechami kierunków turystycznych"
)

plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "korelacje_cech_pastel.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

najlepsze_kultura = (
    df[
        [
            "city",
            "country",
            "culture"
        ]
    ]
    .sort_values(
        by="culture",
        ascending=False
    )
    .head(15)
)

print(
    "\nNajlepsze kierunki pod względem kultury:"
)

print(
    najlepsze_kultura
)


najlepsze_plaze = (
    df[
        [
            "city",
            "country",
            "beaches"
        ]
    ]
    .sort_values(
        by="beaches",
        ascending=False
    )
    .head(15)
)

print(
    "\nNajlepsze kierunki pod względem plaż:"
)

print(
    najlepsze_plaze
)


najlepsze_natura = (
    df[
        [
            "city",
            "country",
            "nature"
        ]
    ]
    .sort_values(
        by="nature",
        ascending=False
    )
    .head(15)
)

print(
    "\nNajlepsze kierunki pod względem natury:"
)

print(
    najlepsze_natura
)

kolumny_temperatur = [
    f"temp_{miesiac}"
    for miesiac in range(1, 13)
]

srednia_temperatura_miesieczna = (
    df[kolumny_temperatur]
    .mean()
)

nazwy_miesiecy = [
    "Styczeń",
    "Luty",
    "Marzec",
    "Kwiecień",
    "Maj",
    "Czerwiec",
    "Lipiec",
    "Sierpień",
    "Wrzesień",
    "Październik",
    "Listopad",
    "Grudzień"
]

plt.figure(figsize=(11, 5))

plt.plot(
    nazwy_miesiecy,
    srednia_temperatura_miesieczna.values,
    marker="o",
    linewidth=2.5,
    color=PASTELOWY_FIOLET,
    markerfacecolor=PASTELOWY_ROZ,
    markeredgecolor=PASTELOWY_FIOLET,
    markersize=7
)

plt.title(
    "Średnia miesięczna temperatura we wszystkich kierunkach"
)

plt.xlabel("Miesiąc")
plt.ylabel("Średnia temperatura [°C]")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "srednia_temperatura_miesieczna_pastel.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()