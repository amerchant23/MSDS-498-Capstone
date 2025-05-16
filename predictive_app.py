import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sklearn
import numpy as np
import joblib
import spacy  # Import spaCy

# Load spaCy model (download if needed: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model (en_core_web_sm)...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

print("Scikit-learn version during training:", sklearn.__version__)
print("NumPy version during training:", np.__version__)


def select_description(x):
    return x['Description']


def clean_description(text):
    text = re.sub(r"<.*?>", "", str(text))  # Remove HTML
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)  # Remove special characters
    return text


def load_and_merge_data(train_file, breed_labels_file, color_labels_file, state_labels_file):
    train_df = pd.read_csv(train_file)
    train_df["Description"] = train_df["Description"].fillna("").astype(str)

    breed_labels_df = pd.read_csv(breed_labels_file)
    color_labels_df = pd.read_csv(color_labels_file)
    state_labels_df = pd.read_csv(state_labels_file)

    train_df = train_df.merge(
        state_labels_df.rename(columns={"StateID": "State"}),
        on="State", how="left"
    ).rename(columns={"StateName": "StateName"})

    train_df = train_df.merge(
        breed_labels_df.rename(columns={"BreedID": "Breed1", "BreedName": "MainBreed", "Type": "Breed1Type"}),
        on="Breed1", how="left"
    )

    train_df = train_df.merge(
        breed_labels_df.rename(columns={"BreedID": "Breed2", "BreedName": "SecondBreed", "Type": "Breed2Type"}),
        on="Breed2", how="left"
    )

    train_df = train_df.merge(
        color_labels_df.rename(columns={"ColorID": "Color1", "ColorName": "ColorName1"}),
        on="Color1", how="left"
    )

    train_df = train_df.merge(
        color_labels_df.rename(columns={"ColorID": "Color2", "ColorName": "ColorName2"}),
        on="Color2", how="left"
    )

    train_df = train_df.merge(
        color_labels_df.rename(columns={"ColorID": "Color3", "ColorName": "ColorName3"}),
        on="Color3", how="left"
    )

    train_df = train_df.drop(columns=["State", "Breed1", "Breed2", "Color1", "Color2", "Color3"])

    train_df["Type"] = train_df["Type"].astype("category").cat.rename_categories({1: "Dog", 2: "Cat"})
    train_df["Gender"] = train_df["Gender"].astype("category").cat.rename_categories({1: "Male", 2: "Female", 3: "Mixed"})
    train_df["AdoptionSpeed"] = train_df["AdoptionSpeed"].astype("category")

    for col in train_df.select_dtypes(include=["object"]).columns:
        if col != "Description":
            train_df[col] = train_df[col].astype("category")

    return train_df


# Load data
train_df = load_and_merge_data("train.csv", "breed_labels.csv", "color_labels.csv", "state_labels.csv")

# Clean descriptions
train_df["Description"] = train_df["Description"].apply(clean_description)

# Feature engineering
train_df["HasName"] = train_df["Name"].apply(lambda x: 0 if pd.isna(x) or x.strip() == "" else 1)
train_df["IsMixBreed"] = train_df["SecondBreed"].notna().astype(int)
train_df["DescriptionLength"] = train_df["Description"].apply(len)

# Define target
y = train_df["AdoptionSpeed"]
X = train_df.drop("AdoptionSpeed", axis=1)

# Categorical column processing
categorical_features = ['Type', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated',
                        'Dewormed', 'Sterilized', 'Health', 'StateName', 'Breed1Type',
                        'Breed2Type', 'MainBreed', 'SecondBreed', 'ColorName1',
                        'ColorName2', 'ColorName3']
for col in categorical_features:
    X[col] = X[col].astype(str)

# Numerical features
numeric_features = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'HasName', 'IsMixBreed', 'DescriptionLength']

# Transformers
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

text_transformer = Pipeline([
    ("selector", FunctionTransformer(select_description, validate=False)),
    ("tfidf", TfidfVectorizer(max_features=100))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
    ("text", text_transformer, ["Description"])
])

# Model pipeline
pipeline_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced"))
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipeline_rf.fit(X_train, y_train)

# Save model and training columns
joblib.dump((pipeline_rf, X_train.columns.tolist()), "pipeline_rf.pkl")
pipeline_rf, saved_columns = joblib.load("pipeline_rf.pkl")


# Evaluate
y_pred_rf = pipeline_rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))


# --- Description Analysis Functions ---

def analyze_description(text):
    """
    Analyzes the description text using spaCy to extract keywords and sentiment.

    Args:
        text (str): The pet description text.

    Returns:
        dict: A dictionary containing the analysis results:
            - keywords (list): A list of important keywords.
            - sentiment (float): A sentiment score (-1 to 1).
    """
    doc = nlp(text)

    # Keyword extraction (simplified - can be improved)
    keywords = [
        token.text
        for token in doc
        if token.is_alpha and not token.is_stop and token.pos_ in {"NOUN", "ADJ"}
    ]

    # Sentiment analysis
    sentiment = doc.sentiment  # Simplified sentiment, requires a trained pipeline.

    return {"keywords": keywords, "sentiment": sentiment}


def generate_suggestions(pet_info, description, analysis_results):
    """
    Generates suggestions for improving the pet description.

    Args:
        pet_info (dict): A dictionary containing the pet's features.
        description (str): The original pet description.
        analysis_results (dict): The results from the analyze_description function.

    Returns:
        list: A list of suggestion strings.
    """
    suggestions = []
    keywords = analysis_results["keywords"]
    sentiment = analysis_results["sentiment"]

    # Suggestion for positive keywords
    positive_keywords = ["friendly", "playful", "loving", "gentle", "loyal"]
    missing_positive_keywords = [
        keyword for keyword in positive_keywords if keyword not in keywords
    ]
    if missing_positive_keywords:
        suggestions.append(
            f"Consider adding positive keywords like: {', '.join(missing_positive_keywords)}"
        )

    # Suggestion for sentiment
    if sentiment < 0.2:  # Arbitrary threshold for negative/neutral sentiment
        suggestions.append(
            "The description could be more positive.  Try to use more enthusiastic language."
        )

    # Suggestion based on pet features.
    if pet_info["Type"] == "Dog":
        if "good with kids" not in description.lower():
            suggestions.append("If the dog is good with children, mention 'good with kids'.")
        if "loves walks" not in description.lower():
            suggestions.append("If the dog enjoys walks, mention 'loves walks'.")
    elif pet_info["Type"] == "Cat":
        if "affectionate" not in description.lower():
            suggestions.append("If the cat is affectionate, mention 'affectionate'.")
        if "clean" not in description.lower():
            suggestions.append("Cats are typically clean, you can mention that.")

    # Suggestion for breed
    if pet_info["MainBreed"]:
        suggestions.append(f"Highlight the positive traits of a {pet_info['MainBreed']}.")

    # Suggestion for age.
    if pet_info["Age"] < 6:
        suggestions.append("Emphasize that this young pet is playful and energetic.")
    elif pet_info["Age"] > 72:
        suggestions.append("Emphasize that this senior pet is calm and loving.")

    return suggestions


def predict_adoption_speed(pet_info: dict, description: str, pipeline: Pipeline, training_columns: list) -> tuple:
    """
    Predicts the adoption speed and generates description suggestions.

    Args:
        pet_info (dict): A dictionary containing the pet's features.
        description (str): The pet description.
        pipeline (Pipeline): The trained prediction pipeline.
        training_columns (list):  The list of columns used during training.

    Returns:
        tuple: (prediction, suggestions)
            - prediction (int): The predicted adoption speed.
            - suggestions (list): A list of description improvement suggestions.
    """
    input_df = pd.DataFrame([pet_info])
    input_df["Description"] = clean_description(description)

    for col in training_columns:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[training_columns]

    # Ensure categorical columns are treated as strings
    categorical_features = ['Type', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated',
                            'Dewormed', 'Sterilized', 'Health', 'StateName', 'Breed1Type',
                            'Breed2Type', 'MainBreed', 'SecondBreed', 'ColorName1',
                            'ColorName2', 'ColorName3']
    for col in categorical_features:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)

    print("Input DataFrame for prediction:")
    print(input_df)
    print("Input DataFrame data types:")
    print(input_df.dtypes)

    try:
        prediction = pipeline.predict(input_df)[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise
    # Only generate suggestions if the predicted speed is 2, 3, or 4
    if prediction in [2, 3, 4]:
        analysis_results = analyze_description(description)  # Analyze description.
        suggestions = generate_suggestions(pet_info, description, analysis_results) # Generate suggestions.
    else:
        suggestions = []
    return prediction, suggestions # Return both prediction and suggestions.


# Example prediction
pet_example = {  # Example pet data
    "Age": 2,
    "Quantity": 1,
    "Fee": 150,
    "VideoAmt": 1,
    "PhotoAmt": 4,
    "Type": "Dog",
    "Gender": "Male",
    "MaturitySize": "Medium",
    "FurLength": "Short",
    "Vaccinated": "Yes",
    "Dewormed": "Yes",
    "Sterilized": "Yes",
    "Health": "Healthy",
    "StateName": "New York",
    "Breed1Type": "Breed",
    "Breed2Type": "Breed",
    "MainBreed": "Golden Retriever",
    "SecondBreed": np.nan,
    "ColorName1": "Golden",
    "ColorName2": "White",
    "ColorName3": "Brown",
    "HasName": 1,
    "IsMixBreed": 0,
    "DescriptionLength": 50,
}
description_example = "A friendly golden retriever looking for a home. Loves to play."

# Predicting
predicted_speed, generated_suggestions = predict_adoption_speed(pet_example, description_example, pipeline_rf, saved_columns)
print("Predicted Adoption Speed:", predicted_speed)
print("Description Suggestions:", generated_suggestions)
