# Netflix Content Type Prediction Model

A machine learning classification model built as part of my machine learning internship project.

The model predicts whether a Netflix title is a **Movie** or **TV Show** based on metadata such as director, rating, release year, added year, added month, and genre information.

## How It Works

The prediction model follows these steps:

1. Explore and clean the Netflix dataset.
2. Select relevant features for classification.
3. Remove unnecessary or unreliable features.
4. Remove `duration` because it directly reveals the content type and would cause target leakage.
5. Convert genre information into numerical features using CountVectorizer.
6. Train a Logistic Regression classification model.
7. Evaluate the model on a stratified test set.
8. Save the complete preprocessing and model pipeline using Joblib.

The model achieved approximately **99.89% accuracy** on the test set.

## Web App

A Streamlit web application is included where users can:

- Enter a Netflix title
- Select the director and rating
- Choose the release year and year added
- Select the month added
- Select one or more genres
- Get a prediction of whether the title is a Movie or TV Show
- View the prediction confidence

The app was also tested with Netflix titles that were not part of the training data.

## Project Structure

Netflix-Content-Type-Prediction-Model/
|
├── Model/
│   └── netflix_content_classifier.pkl
|
├── Notebook/
│   └── Netflix_Content_Type_Prediction_Model.ipynb
|
├── Screenshots/
|
├── app.py
├── README.md
└── requirements.txt

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- CountVectorizer
- Logistic Regression
- Joblib
- Streamlit
- Matplotlib
- Seaborn

## Running the Project

Install the required libraries:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py
