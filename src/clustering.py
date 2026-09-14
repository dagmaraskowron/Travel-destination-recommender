import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "destinations_processed.csv"
IMAGES_DIR = BASE_DIR / "images"

df = pd.read_csv(DATA_PATH)

cechy = [
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

X = df[cechy]

print("Dane używane do klasteryzacji:")
print(X.head())

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nDane po standaryzacji:")
print(X_scaled[:5])

wartosci_k = range(2, 11)

inertia = []
silhouette_scores = []

for k in wartosci_k:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    etykiety = model.fit_predict(X_scaled)

    inertia.append(model.inertia_)

    silhouette_scores.append(
        silhouette_score(X_scaled, etykiety)
    )

plt.figure(figsize=(8, 5))

plt.plot(
    list(wartosci_k),
    inertia,
    marker="o"
)

plt.title("Dobór liczby klastrów – Elbow Method")
plt.xlabel("Liczba klastrów")
plt.ylabel("Wewnątrzklastrowa suma kwadratów")

plt.xticks(list(wartosci_k))

plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "dobor_liczby_klastrow_elbow.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    list(wartosci_k),
    silhouette_scores,
    marker="o"
)

plt.title("Ocena jakości klasteryzacji – Silhouette Score")
plt.xlabel("Liczba klastrów")
plt.ylabel("Współczynnik sylwetki")

plt.xticks(list(wartosci_k))

plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "ocena_klasteryzacji_silhouette.png",
    dpi=300
)

plt.show()

print("\nWyniki Silhouette Score:")

for k, score in zip(wartosci_k, silhouette_scores):
    print(
        f"Liczba klastrów = {k}: "
        f"Silhouette Score = {score:.3f}"
    )

najlepszy_indeks = silhouette_scores.index(
    max(silhouette_scores)
)

najlepsze_k = list(wartosci_k)[najlepszy_indeks]

najlepszy_score = max(silhouette_scores)

print("\nNajlepszy wynik według Silhouette Score:")
print(f"Liczba klastrów: {najlepsze_k}")
print(f"Silhouette Score: {najlepszy_score:.3f}")

wybrane_k = 4

model_kmeans = KMeans(
    n_clusters=wybrane_k,
    random_state=42,
    n_init=10
)

df["cluster"] = model_kmeans.fit_predict(X_scaled)

print("\nLiczba kierunków w poszczególnych klastrach:")

print(
    df["cluster"]
    .value_counts()
    .sort_index()
)


profil_klastrow = (
    df.groupby("cluster")[cechy]
    .mean()
)

print("\nŚrednie wartości cech w klastrach:")

print(
    profil_klastrow.round(2)
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
    "seclusion": "Spokój i odosobnienie"
}

profil_do_wykresu = profil_klastrow.T.copy()

profil_do_wykresu.index = [
    nazwy_cech[cecha]
    for cecha in profil_do_wykresu.index
]

pastelowe_kolory = [
    "#F6C1D1",
    "#CDB4DB",
    "#BDE0FE",
    "#CDEAC0"
]

profil_do_wykresu.plot(
    kind="bar",
    figsize=(12, 7),
    color=pastelowe_kolory
)

plt.title("Profil cech dla poszczególnych klastrów")
plt.xlabel("Cecha")
plt.ylabel("Średnia ocena")
plt.ylim(0, 5)

plt.xticks(rotation=45)

plt.legend(
    title="Klaster"
)

plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "profil_klastrow.png",
    dpi=300
)

plt.show()

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

wariancja = pca.explained_variance_ratio_

print("\nUdział wyjaśnionej wariancji przez PCA:")

print(
    f"Pierwsza składowa: {wariancja[0]:.3f}"
)

print(
    f"Druga składowa: {wariancja[1]:.3f}"
)

print(
    f"Łącznie: {(wariancja[0] + wariancja[1]):.3f}"
)

plt.figure(figsize=(9, 7))

scatter = plt.scatter(
    df["PCA1"],
    df["PCA2"],
    c=df["cluster"],
    alpha=0.7
)

plt.title("Wizualizacja klastrów po redukcji PCA")
plt.xlabel("Pierwsza składowa główna")
plt.ylabel("Druga składowa główna")

plt.colorbar(
    scatter,
    label="Numer klastra"
)

plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "klastry_pca.png",
    dpi=300
)

plt.show()

print("\nPrzykładowe kierunki z każdego klastra:")

for klaster in sorted(df["cluster"].unique()):

    print(f"\nKlaster {klaster}:")

    print(
        df[df["cluster"] == klaster][
            ["city", "country"]
        ].head(10)
    )

nazwy_klastrow = {
    0: "Natura i przygoda",
    1: "Miejskie i kulturowe",
    2: "Spokojne kierunki kulturowe",
    3: "Plaże, natura i relaks"
}

df["cluster_name"] = df["cluster"].map(nazwy_klastrow)

print("\nNazwy klastrów:")
print(
    df[
        ["city", "country", "cluster", "cluster_name"]
    ].head(20)
)

CLUSTERED_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "destinations_clustered.csv"
)

df.to_csv(
    CLUSTERED_PATH,
    index=False
)

print("\nDataset z klastrami zapisano do:")
print(CLUSTERED_PATH)