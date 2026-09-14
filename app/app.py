import pandas as pd
import streamlit as st
from pathlib import Path
from textwrap import dedent

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

st.set_page_config(
    page_title="Travel Destination Recommendation System",
    layout="wide"
)

st.markdown(
    dedent(
        """
        <style>

        .stApp {
            background-color: #FCFAFD;
        }

        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }

        h1, h2, h3 {
            color: #3F3A4D;
        }

        .stButton > button {
            background-color: #D8B4E2;
            color: #2F2936;
            border: none;
            border-radius: 12px;
            padding: 0.7rem 1rem;
            font-weight: 600;
        }

        .stButton > button:hover {
            background-color: #C8A2D6;
            color: #2F2936;
            border: none;
        }

        .recommendation-card {
            background-color: #FFFFFF;
            border: 1px solid #EEE7F1;
            border-radius: 18px;
            padding: 24px;
            min-height: 310px;
            box-shadow: 0 4px 15px rgba(80, 60, 90, 0.07);
            margin-bottom: 10px;
        }

        .recommendation-number {
            font-size: 15px;
            color: #8E8295;
            margin-bottom: 6px;
        }

        .recommendation-city {
            font-size: 27px;
            font-weight: 700;
            color: #3F3A4D;
            margin-bottom: 2px;
        }

        .recommendation-country {
            font-size: 16px;
            color: #706777;
            margin-bottom: 20px;
        }

        .score-box {
            background-color: #F4E9F7;
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 18px;
        }

        .score-label {
            color: #766B7B;
            font-size: 13px;
        }

        .score-value {
            color: #3F3A4D;
            font-size: 30px;
            font-weight: 700;
        }

        .detail-text {
            color: #514B56;
            font-size: 15px;
            line-height: 1.8;
        }

        .why-box {
            background-color: #F7F2FA;
            border-left: 5px solid #CDB4DB;
            border-radius: 10px;
            padding: 18px 20px;
            margin-top: 10px;
            margin-bottom: 20px;
        }

        .info-box {
            background-color: #F1F6FB;
            border-radius: 14px;
            padding: 18px 22px;
            margin-top: 10px;
            margin-bottom: 18px;
        }

        hr {
            border: none;
            border-top: 1px solid #EEE7F1;
        }

        </style>
        """
    ),
    unsafe_allow_html=True
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "destinations_clustered.csv"
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()
miesiace = {
    "Styczeń": 1,
    "Luty": 2,
    "Marzec": 3,
    "Kwiecień": 4,
    "Maj": 5,
    "Czerwiec": 6,
    "Lipiec": 7,
    "Sierpień": 8,
    "Wrzesień": 9,
    "Październik": 10,
    "Listopad": 11,
    "Grudzień": 12
}


budzety = {
    "Budżetowy": 1,
    "Średni": 2,
    "Luksusowy": 3
}


tlumaczenie_budzetu = {
    "Budget": "Budżetowy",
    "Mid-range": "Średni",
    "Luxury": "Luksusowy"
}


dlugosci_wyjazdu = {
    "Jednodniowy": "duration_day_trip",
    "Weekend": "duration_weekend",
    "Krótki wyjazd": "duration_short_trip",
    "Tydzień": "duration_one_week",
    "Długi wyjazd": "duration_long_trip"
}


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

    numer_miesiaca = miesiace[miesiac]
    kolumna_temperatury = f"temp_{numer_miesiaca}"
    kolumna_dlugosci = dlugosci_wyjazdu[dlugosc_wyjazdu]

    classic_scores = []
    feature_scores_all = []
    temperature_scores = []
    budget_scores = []
    duration_scores = []

    for _, row in dane.iterrows():

        wyniki_cech = []

        for cecha, preferowana_wartosc in preferencje.items():

            wynik = score_feature(
                row[cecha],
                preferowana_wartosc
            )

            wyniki_cech.append(wynik)

        score_cech = (
            sum(wyniki_cech)
            / len(wyniki_cech)
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

        classic_score = (
            score_cech * 0.55
            + score_temp * 0.20
            + score_budzet * 0.15
            + score_dlugosc * 0.10
        )

        classic_scores.append(classic_score)
        feature_scores_all.append(score_cech)
        temperature_scores.append(score_temp)
        budget_scores.append(score_budzet)
        duration_scores.append(score_dlugosc)

    dane["classic_score"] = classic_scores
    dane["feature_score"] = feature_scores_all
    dane["temperature_score"] = temperature_scores
    dane["budget_score"] = budget_scores
    dane["duration_score"] = duration_scores

    return dane

def create_destination_description(row):

    return (
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

@st.cache_resource
def load_nlp_model_and_embeddings():

    model = SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    descriptions = df.apply(
        create_destination_description,
        axis=1
    ).tolist()

    embeddings = model.encode(
        descriptions,
        show_progress_bar=False
    )

    return model, embeddings

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

    model, destination_embeddings = (
        load_nlp_model_and_embeddings()
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

    if max_nlp != min_nlp:

        results["nlp_score_normalized"] = (
            results["nlp_score"] - min_nlp
        ) / (
            max_nlp - min_nlp
        )

    else:

        results["nlp_score_normalized"] = 0

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

def create_explanation(
    row,
    preferencje,
    wybrany_miesiac,
    preferowana_temperatura
):

    reasons = []

    numer_miesiaca = miesiace[wybrany_miesiac]

    temp = row[f"temp_{numer_miesiaca}"]

    roznica_temp = abs(
        temp - preferowana_temperatura
    )

    if roznica_temp <= 2:

        reasons.append(
            f"Temperatura {temp:.1f}°C jest bardzo bliska "
            f"Twojej preferowanej temperaturze "
            f"{preferowana_temperatura}°C."
        )

    elif roznica_temp <= 5:

        reasons.append(
            f"Temperatura {temp:.1f}°C dobrze odpowiada "
            f"Twoim preferencjom."
        )

    if row["budget_score"] == 1:

        reasons.append(
            "Poziom cenowy kierunku odpowiada wybranemu budżetowi."
        )

    elif row["budget_score"] == 0.5:

        reasons.append(
            "Poziom cenowy jest zbliżony do wybranego budżetu."
        )

    if row["duration_score"] == 1:

        reasons.append(
            "Kierunek jest odpowiedni dla wybranej długości podróży."
        )

    dopasowania = []

    for cecha, preferowana_wartosc in preferencje.items():

        wynik = score_feature(
            row[cecha],
            preferowana_wartosc
        )

        dopasowania.append(
            (
                cecha,
                wynik,
                row[cecha]
            )
        )

    dopasowania.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_cechy = dopasowania[:3]

    tekst_cech = ", ".join(
        [
            f"{nazwy_cech[cecha]} ({int(wartosc)}/5)"
            for cecha, _, wartosc in top_cechy
        ]
    )

    reasons.append(
        f"Najlepiej dopasowane cechy kierunku: {tekst_cech}."
    )

    if row["nlp_match_percent"] >= 55:

        reasons.append(
            "Opis kierunku jest również wysoko zgodny "
            "z tekstowym opisem Twojego idealnego wyjazdu."
        )

    return reasons

st.title("Travel Destination Recommendation System")

st.write(
    """
    System rekomenduje kierunki podróży na podstawie preferencji,
    warunków pogodowych, poziomu budżetu, długości wyjazdu oraz
    analizy semantycznej opisu idealnej podróży.
    """
)

st.divider()

st.header("Parametry podróży")

col1, col2, col3, col4 = st.columns(4)


with col1:

    wybrany_miesiac = st.selectbox(
        "Miesiąc podróży",
        list(miesiace.keys()),
        index=8
    )


with col2:

    preferowana_temperatura = st.slider(
        "Preferowana temperatura [°C]",
        min_value=-10,
        max_value=40,
        value=26
    )


with col3:

    wybrany_budzet = st.selectbox(
        "Poziom budżetu",
        list(budzety.keys()),
        index=1
    )


with col4:

    wybrana_dlugosc = st.selectbox(
        "Długość wyjazdu",
        list(dlugosci_wyjazdu.keys()),
        index=3
    )

st.header("Twoje preferencje")

st.write(
    "Oceń poszczególne elementy podróży w skali od 1 do 5."
)


col1, col2, col3 = st.columns(3)


with col1:

    kultura = st.slider(
        "Kultura",
        1,
        5,
        3
    )

    przygoda = st.slider(
        "Przygoda",
        1,
        5,
        3
    )

    natura = st.slider(
        "Natura",
        1,
        5,
        4
    )


with col2:

    plaze = st.slider(
        "Plaże",
        1,
        5,
        4
    )

    zycie_nocne = st.slider(
        "Życie nocne",
        1,
        5,
        3
    )

    kuchnia = st.slider(
        "Kuchnia",
        1,
        5,
        4
    )


with col3:

    wellness = st.slider(
        "Wellness",
        1,
        5,
        3
    )

    miejski = st.slider(
        "Miejski charakter",
        1,
        5,
        3
    )

    spokoj = st.slider(
        "Spokój i odosobnienie",
        1,
        5,
        3
    )


preferencje_uzytkownika = {
    "culture": kultura,
    "adventure": przygoda,
    "nature": natura,
    "beaches": plaze,
    "nightlife": zycie_nocne,
    "cuisine": kuchnia,
    "wellness": wellness,
    "urban": miejski,
    "seclusion": spokoj
}

st.header("Opisz swój idealny wyjazd")


user_text = st.text_area(
    "Opisz, czego szukasz:",
    value=(
        "Chcę pojechać w ciepłe miejsce nad morzem. "
        "Zależy mi na pięknych plażach, naturze i dobrej kuchni. "
        "Lubię trochę zwiedzać, ale wolę spokojniejszy wyjazd "
        "niż intensywne życie nocne."
    ),
    height=130
)

st.divider()


if st.button(
    "Znajdź najlepsze kierunki",
    type="primary",
    use_container_width=True
):

    if len(user_text.strip()) < 10:

        st.warning(
            "Opis idealnego wyjazdu powinien być trochę dłuższy."
        )

    else:

        with st.spinner(
            "Analizuję kierunki i przygotowuję rekomendacje..."
        ):

            wyniki = hybrid_recommendation(
                dane=df,
                user_text=user_text,
                miesiac=wybrany_miesiac,
                preferowana_temperatura=preferowana_temperatura,
                budzet=budzety[wybrany_budzet],
                dlugosc_wyjazdu=wybrana_dlugosc,
                preferencje=preferencje_uzytkownika,
                classic_weight=0.7,
                nlp_weight=0.3,
                top_n=10
            )


        st.success(
            "Rekomendacje zostały przygotowane."
        )

        st.header("Najlepsze rekomendacje")


        top3 = wyniki.head(3)

        kolumny = st.columns(3)

        numer_miesiaca = miesiace[
            wybrany_miesiac
        ]


        for i, (_, row) in enumerate(
            top3.iterrows()
        ):

            temperatura = row[
                f"temp_{numer_miesiaca}"
            ]

            budzet_pl = tlumaczenie_budzetu.get(
                row["budget_level"],
                row["budget_level"]
            )


            with kolumny[i]:

                karta_html = dedent(
                    f"""
                    <div class="recommendation-card">
                        <div class="recommendation-number">
                            Rekomendacja {i + 1}
                        </div>

                        <div class="recommendation-city">
                            {row['city']}
                        </div>

                        <div class="recommendation-country">
                            {row['country']}
                        </div>

                        <div class="score-box">
                            <div class="score-label">
                                Dopasowanie
                            </div>

                            <div class="score-value">
                                {row['hybrid_match_percent']:.1f}%
                            </div>
                        </div>

                        <div class="detail-text">
                            <b>Typ kierunku:</b><br>
                            {row['cluster_name']}<br><br>

                            <b>Budżet:</b><br>
                            {budzet_pl}<br><br>

                            <b>Temperatura:</b><br>
                            {temperatura:.1f}°C
                        </div>
                    </div>
                    """
                )

                st.markdown(
                    karta_html,
                    unsafe_allow_html=True
                )

        st.header("Dlaczego te kierunki?")


        for numer, (_, row) in enumerate(
            top3.iterrows(),
            start=1
        ):

            reasons = create_explanation(
                row=row,
                preferencje=preferencje_uzytkownika,
                wybrany_miesiac=wybrany_miesiac,
                preferowana_temperatura=preferowana_temperatura
            )


            st.subheader(
                f"{numer}. {row['city']}, {row['country']}"
            )


            lista_powodow = "".join(
                f"<li>{powod}</li>"
                for powod in reasons
            )


            explanation_html = dedent(
                f"""
                <div class="why-box">
                    <ul>
                        {lista_powodow}
                    </ul>
                </div>
                """
            )


            st.markdown(
                explanation_html,
                unsafe_allow_html=True
            )

        st.header("TOP 10 kierunków")


        tabela = wyniki[
            [
                "city",
                "country",
                "cluster_name",
                "budget_level",
                "classic_match_percent",
                "nlp_match_percent",
                "hybrid_match_percent"
            ]
        ].copy()


        tabela["budget_level"] = (
            tabela["budget_level"]
            .map(tlumaczenie_budzetu)
        )


        tabela.columns = [
            "Miasto",
            "Kraj",
            "Typ kierunku",
            "Budżet",
            "Dopasowanie klasyczne",
            "Dopasowanie NLP",
            "Wynik końcowy"
        ]


        tabela["Dopasowanie klasyczne"] = (
            tabela["Dopasowanie klasyczne"]
            .map(lambda x: f"{x:.1f}%")
        )


        tabela["Dopasowanie NLP"] = (
            tabela["Dopasowanie NLP"]
            .map(lambda x: f"{x:.1f}%")
        )


        tabela["Wynik końcowy"] = (
            tabela["Wynik końcowy"]
            .map(lambda x: f"{x:.1f}%")
        )


        st.dataframe(
            tabela,
            use_container_width=True,
            hide_index=True
        )

        st.header("Porównanie dopasowania")


        wykres_df = wyniki[
            [
                "city",
                "hybrid_match_percent"
            ]
        ].copy()


        wykres_df = wykres_df.sort_values(
            by="hybrid_match_percent",
            ascending=True
        )


        fig = px.bar(
            wykres_df,
            x="hybrid_match_percent",
            y="city",
            orientation="h",
            labels={
                "hybrid_match_percent": "Dopasowanie [%]",
                "city": "Kierunek"
            }
        )


        fig.update_traces(
            marker_color="#CDB4DB"
        )


        fig.update_layout(
            title="Końcowe dopasowanie rekomendowanych kierunków",
            plot_bgcolor="#FCFAFD",
            paper_bgcolor="#FCFAFD",
            font=dict(
                color="#3F3A4D"
            ),
            xaxis_title="Dopasowanie [%]",
            yaxis_title="",
            showlegend=False
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.header("Polecane kierunki na mapie")


        mapa = wyniki[
            [
                "latitude",
                "longitude"
            ]
        ].copy()


        st.map(
            mapa,
            use_container_width=True
        )

        najlepszy = wyniki.iloc[0]


        st.header(
            f"Więcej o kierunku: {najlepszy['city']}"
        )


        budzet_najlepszego = tlumaczenie_budzetu.get(
            najlepszy["budget_level"],
            najlepszy["budget_level"]
        )


        info_html = dedent(
            f"""
            <div class="info-box">
                <b>Kraj:</b>
                {najlepszy['country']}
                <br><br>

                <b>Typ kierunku:</b>
                {najlepszy['cluster_name']}
                <br><br>

                <b>Poziom budżetu:</b>
                {budzet_najlepszego}
                <br><br>

                <b>Opis źródłowy:</b>
                <br>
                {najlepszy['short_description']}
            </div>
            """
        )


        st.markdown(
            info_html,
            unsafe_allow_html=True
        )

        st.subheader(
            "Profil najlepszego kierunku"
        )


        oceny = pd.DataFrame(
            {
                "Cecha": [
                    nazwy_cech[cecha]
                    for cecha in nazwy_cech
                ],

                "Ocena": [
                    najlepszy[cecha]
                    for cecha in nazwy_cech
                ]
            }
        )


        fig_oceny = px.bar(
            oceny,
            x="Ocena",
            y="Cecha",
            orientation="h",
            range_x=[0, 5]
        )


        fig_oceny.update_traces(
            marker_color="#BDE0FE"
        )


        fig_oceny.update_layout(
            plot_bgcolor="#FCFAFD",
            paper_bgcolor="#FCFAFD",
            font=dict(
                color="#3F3A4D"
            ),
            xaxis_title="Ocena",
            yaxis_title="",
            showlegend=False
        )


        st.plotly_chart(
            fig_oceny,
            use_container_width=True
        )

        with st.expander(
            "Jak obliczane jest dopasowanie?"
        ):

            st.write(
                """
                Końcowy ranking jest tworzony przez hybrydowy
                system rekomendacyjny.

                70% wyniku pochodzi z klasycznego modelu
                uwzględniającego preferencje użytkownika,
                temperaturę, budżet oraz długość wyjazdu.

                30% wyniku pochodzi z modelu NLP, który porównuje
                tekstowy opis idealnej podróży z opisami kierunków
                za pomocą embeddingów.

                Przed połączeniem obu wyników podobieństwo NLP jest
                normalizowane do zakresu od 0 do 1.
                """
            )


st.divider()

st.caption(
    "Travel Destination Recommendation System"
)