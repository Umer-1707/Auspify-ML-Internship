# Netflix Content Recommendation System

A content-based Netflix recommendation system built as part of my machine learning internship project.

The system recommends similar movies and TV shows based mainly on genre and category information, along with other content features such as type, rating, and duration.

## How It Works

The recommendation system follows these steps:

1. Prepare content-related features from the Netflix dataset.
2. Combine the selected features into a single text representation.
3. Convert the text into numerical features using TF-IDF.
4. Calculate similarity between titles using cosine similarity.
5. Return the most similar titles for a selected movie or TV show.

Since the dataset does not contain user ratings or viewing history, the system uses content similarity rather than collaborative filtering.

## Web App

A Streamlit web application is included where users can:

- Search and select a Netflix title
- Choose the number of recommendations
- View similar movies or TV shows
- See genre, type, rating, duration, and similarity score for each recommendation

## Project Structure

Netflix-Content-Recommendation-System/

│
├── Model-and-Helpers/
│   ├── cosine_similarity.pkl
│   ├── tfidf_vectorizer.pkl
│   └── title_lookup.pkl
│
├── Notebook/
│   └── Netflix_Content_Recommendation_System.ipynb
│
├── Recommendation-Dataset/
│   └── netflix_recommendation_data.csv
│
├── Screenshots/
│
├── app.py
├── requirements.txt
└── README.md

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Joblib
- Streamlit
- Matplotlib
- Seaborn

## Running the Project

Install the required libraries:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

## Note

The recommendation model is trained using the Netflix-style dataset. The similarity matrix and other required model files are saved separately so the Streamlit app can load them directly without retraining the model.