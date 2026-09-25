import streamlit as st
import pandas as pd
import joblib

def split_genres_func(text):
    return text.split(", ")

model = joblib.load("Model/netflix_content_classifier.pkl")

st.markdown("""
<style>
    .stApp {
        background-color: #111111;
        color: white;
    }

    h1 {
        color: #E50914;
    }

    .stButton > button {
        background-color: #E50914;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #B20710;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Netflix Content Type Predictor",
    page_icon="🎬",
    layout="centered"
)


st.title("Netflix Content Type Predictor")
st.write(
    "Predict whether a Netflix title is a Movie or TV Show "
    "based on its metadata."
)

st.divider()


# Get the values learned by the encoders
preprocessor = model.named_steps["preprocessor"]

director_encoder = preprocessor.named_transformers_["categorical"]
directors = director_encoder.categories_[0]
ratings = director_encoder.categories_[1]

genres_vectorizer = preprocessor.named_transformers_["genres"]
genres = genres_vectorizer.get_feature_names_out()


st.subheader("Title Information")

title = st.text_input("Title")


director = st.selectbox(
    "Director",
    directors
)


rating = st.selectbox(
    "Rating",
    ratings
)


release_years = list(range(1925, 2027))
added_years = list(range(2008, 2027))

release_year = st.selectbox(
    "Release Year",
    release_years,
    index=release_years.index(2018)
)

year_added = st.selectbox(
    "Year Added",
    added_years,
    index=added_years.index(2018)
)


month_names = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

month_added = st.selectbox(
    "Month Added",
    month_names
)

month_number = month_names.index(month_added) + 1


selected_genres = st.multiselect(
    "Genres",
    genres
)


st.divider()


if st.button("Predict", use_container_width=True):

    if not title.strip():
        st.warning("Please enter a title.")
    elif not selected_genres:
        st.warning("Please select at least one genre.")
    else:

        genre_text = ", ".join(selected_genres)

        input_data = pd.DataFrame([{
            "director": director,
            "rating": rating,
            "release_year": release_year,
            "year_added": year_added,
            "month_added": month_number,
            "listed_in": genre_text
        }])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        class_probabilities = dict(
            zip(model.classes_, probabilities)
        )

        confidence = class_probabilities[prediction]

        st.subheader("Prediction")

        if prediction == "Movie":
            st.success("MOVIE")
        else:
            st.info("TV SHOW")

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )