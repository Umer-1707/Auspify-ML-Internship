import streamlit as st
import pandas as pd
import joblib


# Page settings

st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# Color palette

BG = "#081B30"
DARK_TEAL = "#003951"
PURPLE = "#441D36"
CYAN = "#00DBCA"
RED = "#BD000F"
BURGUNDY = "#4C0519"


# Custom styling

st.markdown(
    f"""
<style>

.stApp {{
    background: linear-gradient(
        135deg,
        {BG} 0%,
        {DARK_TEAL} 55%,
        {BURGUNDY} 100%
    );
    color: white;
}}

h1 {{
    color: {CYAN};
    font-size: 42px !important;
    margin-bottom: 5px;
}}

.description {{
    color: #c7d4dc;
    font-size: 17px;
    margin-bottom: 30px;
}}

.section-title {{
    color: {CYAN};
    font-size: 20px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 10px;
}}

.selected-title {{
    background: rgba(68, 29, 54, 0.8);
    border-left: 4px solid {CYAN};
    border-radius: 8px;
    padding: 15px 18px;
    margin: 20px 0;
}}

div[data-baseweb="select"] > div {{
    background-color: {DARK_TEAL};
    border-color: {CYAN};
}}

div[data-baseweb="select"] span {{
    color: white;
}}

.footer {{
    text-align: center;
    color: #8da3af;
    font-size: 13px;
    margin-top: 45px;
    padding-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}}

</style>
""",
    unsafe_allow_html=True
)


# Load saved files

@st.cache_data
def load_data():

    df = pd.read_csv(
        "Recommendation-Dataset/netflix_recommendation_data.csv"
    )

    cosine_sim = joblib.load(
        "Model-and-Helpers/cosine_similarity.pkl"
    )

    title_lookup = joblib.load(
        "Model-and-Helpers/title_lookup.pkl"
    )

    return df, cosine_sim, title_lookup


df, cosine_sim, title_lookup = load_data()


# Header

st.title("Netflix Content Recommendation System")

st.markdown(
    f"""
    <div class="description">
        Discover Netflix titles similar to your favorite movies and TV shows.
        Recommendations are based on genres, categories, type, rating, and duration.
    </div>
    """,
    unsafe_allow_html=True
)


# Title selection

st.markdown(
    '<div class="section-title">Choose a title</div>',
    unsafe_allow_html=True
)

titles = sorted(
    df["title"].dropna().unique()
)

selected_title = st.selectbox(
    "Search for a movie or TV show",
    titles,
    index=None,
    placeholder="Start typing a title..."
)


# Number of recommendations

st.markdown(
    '<div class="section-title">Number of recommendations</div>',
    unsafe_allow_html=True
)

num_recommendations = st.selectbox(
    "How many recommendations would you like?",
    options=list(range(0, 11)),
    index=5
)


if selected_title:

    idx = title_lookup.get(selected_title.lower())

    if idx is not None:

        selected_row = df.iloc[idx]

        st.markdown(
            f"""
            <div class="selected-title">
                <strong>Selected:</strong> {selected_row["title"]}
                &nbsp; | &nbsp;
                {selected_row["type"]}
            </div>
            """,
            unsafe_allow_html=True
        )


        # Calculate recommendations

        similarity_scores = list(
            enumerate(cosine_sim[idx])
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        similarity_scores = similarity_scores[
            1:num_recommendations + 1
        ]

        recommendation_indices = [
            item[0]
            for item in similarity_scores
        ]

        recommendations = df.iloc[
            recommendation_indices
        ].copy()


        # Recommendation heading

        st.markdown(
            '<div class="section-title">Recommended for you</div>',
            unsafe_allow_html=True
        )


        # Display recommendations

        for i, ((_, row), (_, score)) in enumerate(
            zip(
                recommendations.iterrows(),
                similarity_scores
            ),
            start=1
        ):

            with st.container(border=True):

                st.subheader(
                    f"{i}. {row['title']}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(
                        f"**Type:** {row['type']}"
                    )

                with col2:
                    st.write(
                        f"**Rating:** {row['rating']}"
                    )

                with col3:
                    st.write(
                        f"**Duration:** {row['duration']}"
                    )

                st.write(
                    f"**Genres:** {row['listed_in']}"
                )

                st.caption(
                    f"Similarity Score: {score:.3f}"
                )

    else:

        st.warning(
            "Title not found in the dataset."
        )


# Footer

st.markdown(
    """
    <div class="footer">
        Content-Based Recommendation System · TF-IDF + Cosine Similarity
    </div>
    """,
    unsafe_allow_html=True
)