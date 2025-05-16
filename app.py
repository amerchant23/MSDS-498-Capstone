import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import requests
import os
import traceback  # Import the traceback module for detailed error logging

# --- MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="🐾 Pet Adoption Predictor", layout="wide")

# --- Configuration ---
MODEL_URL = "https://drive.google.com/file/d/1k-PsQJTdXjuCVQSStvdua04OkEmb2hr4/view?usp=sharing"
MODEL_FILENAME = "pipeline_rf.pkl"
LOCAL_MODEL_PATH = MODEL_FILENAME

# Utility functions (clean_description, analyze_description, generate_suggestions, predict_adoption_speed)
def clean_description(text):
    text = re.sub(r"<.*?>", "", str(text))
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)
    return text

def analyze_description(text):
    keywords = re.findall(r"[a-zA-Z0-9]+", text.lower())
    stop_words = set(['the', 'a', 'is', 'in', 'it', 'and', 'of', 'to', 'be', 'with', 'for'])
    filtered_keywords = [word for word in keywords if word not in stop_words and len(word) > 2]
    return {"keywords": filtered_keywords, "sentiment": 0.0}

def generate_suggestions(pet_info, description, analysis_results):
    suggestions = []
    keywords = analysis_results["keywords"]
    sentiment = analysis_results["sentiment"]
    positive_keywords = ["friendly", "playful", "loving", "gentle", "loyal", "sweet", "happy"]
    missing_positive_keywords = [
        keyword for keyword in positive_keywords if keyword not in keywords
    ]
    if missing_positive_keywords:
        suggestions.append(
            f"Consider adding positive words like: {', '.join(missing_positive_keywords)}"
        )
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
    if pet_info["MainBreed"]:
        breed_lower = pet_info["MainBreed"].lower()
        if breed_lower not in description.lower():
            suggestions.append(f"Consider highlighting the breed: {pet_info['MainBreed']}.")
    if pet_info["Age"] < 6:
        suggestions.append("Emphasize that this young pet is playful and energetic.")
    elif pet_info["Age"] > 72:
        suggestions.append("Emphasize that this senior pet is calm and loving.")
    return suggestions

def predict_adoption_speed(pet_info: dict, description: str, pipeline, training_columns: list) -> tuple:
    input_df = pd.DataFrame([pet_info])
    input_df["Description"] = clean_description(description)

    if training_columns is None:
        st.error("Error: training_columns is None. Please check model loading.")
        return None, []

    for col in training_columns:
        if col not in input_df.columns:
            input_df[col] = np.nan

    categorical_features = ['Type', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated',
                            'Dewormed', 'Sterilized', 'Health', 'StateName', 'Breed1Type',
                            'Breed2Type', 'MainBreed', 'SecondBreed', 'ColorName1',
                            'ColorName2', 'ColorName3']
    for col in categorical_features:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)

    try:
        prediction = pipeline.predict(input_df)[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, []

    if prediction in [2, 3, 4]:
        analysis_results = analyze_description(description)
        suggestions = generate_suggestions(pet_info, description, analysis_results)
    else:
        suggestions = []
    return prediction, suggestions

@st.cache_resource
def load_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.info(f"Downloading model from {MODEL_URL}...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(LOCAL_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model downloaded successfully!")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}")
            return None, None  # Return None for both in case of download error
        except Exception as e:
            st.error(f"An unexpected error occurred during download: {e}")
            return None, None  # Return None for both in case of other download errors

    try:
        pipeline_rf, saved_columns = joblib.load(LOCAL_MODEL_PATH)
        return pipeline_rf, saved_columns
    except Exception as e:
        st.error(f"Error loading the model from {LOCAL_MODEL_PATH}: {e}")
        st.error(traceback.format_exc())  # Print the full traceback for loading errors
        return None, None

# --- Load the model ---
model_tuple = load_model()
if model_tuple:
    pipeline_rf, saved_columns = model_tuple
else:
    st.stop()

# Breed lists for Dogs and Cats
dog_breeds = [
    "", "Affenpinscher", "Afghan Hound", "Airedale Terrier", "Akbash", "Akita", "Alaskan Malamute",
    "American Bulldog", "American Eskimo Dog", "American Hairless Terrier", "American Staffordshire Terrier",
    "American Water Spaniel", "Anatolian Shepherd", "Appenzell Mountain Dog", "Australian Cattle Dog/Blue Heeler",
    "Australian Kelpie", "Australian Shepherd", "Australian Terrier", "Basenji", "Basset Hound", "Beagle",
    "Bearded Collie", "Beauceron", "Bedlington Terrier", "Belgian Shepherd Dog Sheepdog",
    "Belgian Shepherd Laekenois", "Belgian Shepherd Malinois", "Belgian Shepherd Tervuren",
    "Bernese Mountain Dog", "Bichon Frise", "Black and Tan Coonhound", "Black Labrador Retriever",
    "Black Mouth Cur", "Black Russian Terrier", "Bloodhound", "Blue Lacy", "Bluetick Coonhound", "Boerboel",
    "Bolognese", "Border Collie", "Border Terrier", "Borzoi", "Boston Terrier", "Bouvier des Flanders", "Boxer",
    "Boykin Spaniel", "Briard", "Brittany Spaniel", "Brussels Griffon", "Bull Terrier", "Bullmastiff",
    "Cairn Terrier", "Canaan Dog", "Cane Corso Mastiff", "Carolina Dog", "Catahoula Leopard Dog", "Cattle Dog",
    "Caucasian Sheepdog (Caucasian Ovtcharka)", "Cavalier King Charles Spaniel", "Chesapeake Bay Retriever",
    "Chihuahua", "Chinese Crested Dog", "Chinese Foo Dog", "Chinook", "Chocolate Labrador Retriever", "Chow Chow",
    "Cirneco dell'Etna", "Clumber Spaniel", "Cockapoo", "Cocker Spaniel", "Collie", "Coonhound", "Corgi",
    "Coton de Tulear", "Curly-Coated Retriever", "Dachshund", "Dalmatian", "Dandi Dinmont Terrier",
    "Doberman Pinscher", "Dogo Argentino", "Dogue de Bordeaux", "Dutch Shepherd", "English Bulldog",
    "English Cocker Spaniel", "English Coonhound", "English Pointer", "English Setter", "English Shepherd",
    "English Springer Spaniel", "English Toy Spaniel", "Entlebucher", "Eskimo Dog", "Feist", "Field Spaniel",
    "Fila Brasileiro", "Finnish Lapphund", "Finnish Spitz", "Flat-coated Retriever", "Fox Terrier", "Foxhound",
    "French Bulldog", "Galgo Spanish Greyhound", "German Pinscher", "German Shepherd Dog",
    "German Shorthaired Pointer", "German Spitz", "German Wirehaired Pointer", "Giant Schnauzer",
    "Glen of Imaal Terrier", "Golden Retriever", "Gordon Setter", "Great Dane", "Great Pyrenees",
    "Greater Swiss Mountain Dog", "Greyhound", "Harrier", "Havanese", "Hound", "Hovawart", "Husky",
    "Ibizan Hound", "Illyrian Sheepdog", "Irish Setter", "Irish Terrier", "Irish Water Spaniel",
    "Irish Wolfhound", "Italian Greyhound", "Italian Spinone", "Jack Russell Terrier",
    "Jack Russell Terrier (Parson Russell Terrier)", "Japanese Chin", "Jindo", "Kai Dog",
    "Karelian Bear Dog", "Keeshond", "Kerry Blue Terrier", "Kishu", "Klee Kai", "Komondor", "Kuvasz",
    "Kyi Leo", "Labrador Retriever", "Lakeland Terrier", "Lancashire Heeler", "Leonberger", "Lhasa Apso",
    "Lowchen", "Maltese", "Manchester Terrier", "Maremma Sheepdog", "Mastiff", "McNab", "Miniature Pinscher",
    "Mountain Cur", "Mountain Dog", "Munsterlander", "Neapolitan Mastiff", "New Guinea Singing Dog",
    "Newfoundland Dog", "Norfolk Terrier", "Norwegian Buhund", "Norwegian Elkhound", "Norwegian Lundehund",
    "Norwich Terrier", "Nova Scotia Duck-Tolling Retriever", "Old English Sheepdog", "Otterhound", "Papillon",
    "Patterdale Terrier (Fell Terrier)", "Pekingese", "Peruvian Inca Orchid", "Petit Basset Griffon Vendeen",
    "Pharaoh Hound", "Pit Bull Terrier", "Plott Hound", "Podengo Portugueso", "Pointer",
    "Polish Lowland Sheepdog", "Pomeranian", "Poodle", "Portuguese Water Dog", "Presa Canario", "Pug", "Puli",
    "Pumi", "Rat Terrier", "Redbone Coonhound", "Retriever", "Rhodesian Ridgeback", "Rottweiler",
    "Saint Bernard", "Saluki", "Samoyed", "Sarplaninac", "Schipperke", "Schnauzer", "Scottish Deerhound",
    "Scottish Terrier Scottie", "Sealyham Terrier", "Setter", "Shar Pei", "Sheep Dog", "Shepherd",
    "Shetland Sheepdog Sheltie", "Shiba Inu", "Shih Tzu", "Siberian Husky", "Silky Terrier", "Skye Terrier",
    "Sloughi", "Smooth Fox Terrier", "South Russian Ovtcharka", "Spaniel", "Spitz", "Staffordshire Bull Terrier",
    "Standard Poodle", "Sussex Spaniel", "Swedish Vallhund", "Terrier", "Thai Ridgeback", "Tibetan Mastiff",
    "Tibetan Spaniel", "Tibetan Terrier", "Tosa Inu", "Toy Fox Terrier", "Treeing Walker Coonhound", "Vizsla",
    "Weimaraner", "Welsh Corgi", "Welsh Springer Spaniel", "Welsh Terrier",
    "West Highland White Terrier Westie", "Wheaten Terrier", "Whippet", "White German Shepherd",
    "Wire Fox Terrier", "Wire-haired Pointing Griffon", "Wirehaired Terrier", "Xoloitzcuintle/Mexican Hairless",
    "Yellow Labrador Retriever", "Yorkshire Terrier Yorkie", "Mixed Breed"
]

cat_breeds = [
    "", "Abyssinian", "American Curl", "American Shorthair", "American Wirehair", "Applehead Siamese", "Balinese",
    "Bengal", "Birman", "Bobtail", "Bombay", "British Shorthair", "Burmese", "Burmilla", "Calico",
    "Canadian Hairless", "Chartreux", "Chausie", "Chinchilla", "Cornish Rex", "Cymric", "Devon Rex",
    "Dilute Calico", "Dilute Tortoiseshell", "Domestic Long Hair", "Domestic Medium Hair", "Domestic Short Hair",
    "Egyptian Mau", "Exotic Shorthair", "Extra-Toes Cat (Hemingway Polydactyl)", "Havana", "Himalayan",
    "Japanese Bobtail", "Javanese", "Korat", "LaPerm", "Maine Coon", "Manx", "Munchkin", "Nebelung",
    "Norwegian Forest Cat", "Ocicat", "Oriental Long Hair", "Oriental Short Hair", "Oriental Tabby", "Persian",
    "Pixie-Bob", "Ragamuffin", "Ragdoll", "Russian Blue", "Scottish Fold", "Selkirk Rex", "Siamese", "Siberian",
    "Silver", "Singapura", "Snowshoe", "Somali", "Sphynx (hairless cat)", "Tabby", "Tiger", "Tonkinese", "Torbie",
    "Tortoiseshell", "Turkish Angora", "Turkish Van"
]

# Specific color options
color_options_list = ['', 'Black', 'Brown', 'Golden', 'Yellow', 'Cream', 'Gray', 'White']

# Specific state options
state_options_list = ['', 'Johor', 'Kedah', 'Kelantan', 'Kuala Lumpur', 'Labuan', 'Melaka',
                      'Negeri Sembilan', 'Pahang', 'Perak', 'Perlis', 'Pulau Pinang',
                      'Sabah', 'Sarawak', 'Selangor', 'Terengganu']

# --- App UI ---
st.title("🐶 Pet Adoption Speed Predictor")
st.write("Predict how quickly a pet may be adopted based on its characteristics and description.")

with st.sidebar:
    st.header("📋 Pet Profile")

    st.subheader("🐾 Basic Info")
    age = st.number_input("Age (months)", 0, 240)
    quantity = st.number_input("Quantity (Pet Count in Profile)", 1, 20)
    fee = st.number_input("Adoption Fee ($)", 0, 1000)
    video_amt = st.number_input("Number of Videos", 0, 10)
    photo_amt = st.number_input("Number of Photos", 0, 20)

    st.subheader("🧬 Biological Info")
    type = st.selectbox("Type", ["Dog", "Cat"])
    gender = st.selectbox("Gender", ["Male", "Female", "Mixed"])
    maturity_size = st.selectbox("Maturity Size", ["Small", "Medium", "Large", "Extra Large"])
    fur_length = st.selectbox("Fur Length", ["Short", "Medium", "Long"])
    vaccinated = st.selectbox("Vaccinated", ["Yes", "No", "Not Sure"])
    dewormed = st.selectbox("Dewormed", ["Yes", "No", "Not Sure"])
    sterilized = st.selectbox("Sterilized", ["Yes", "No", "Not Sure"])
    health = st.selectbox("Health", ["Healthy", "Minor Injury", "Serious Injury"])
    state_name = st.selectbox("State", state_options_list)

    st.subheader("🐕 Breed Info")
    if type == "Dog":
        main_breed_options = dog_breeds
        second_breed_options = dog_breeds
    elif type == "Cat":
        main_breed_options = cat_breeds
        second_breed_options = cat_breeds
    else:
        main_breed_options = [""]
        second_breed_options = [""]

    main_breed = st.selectbox("Main Breed", main_breed_options)
    second_breed = st.selectbox("Second Breed (if any)", second_breed_options)

    breed1_type = "Breed" if main_breed else np.nan
    breed2_type = "Breed" if second_breed else np.nan

    st.subheader("🎨 Colors")
    color_name1 = st.selectbox("Color 1", color_options_list)
    color_name2 = st.selectbox("Color 2 (optional)", color_options_list)
    color_name3 = st.selectbox("Color 3 (optional)", color_options_list)

    st.subheader("📛 Optional")
    pet_name = st.text_input("Pet Name (optional)", key="pet_name")

# Main area: Description and Prediction
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Pet Description")
    description = st.text_area("Describe the pet (personality, temperament, etc.):", height=150)

with col2:
    st.subheader("🔮 Prediction")
    if st.button("Predict Adoption Speed"):
        if not description.strip():
            st.warning("Please enter a description.")
        else:
            maturity_mapping = {"Small": 1, "Medium": 2, "Large": 3, "Extra Large": 4}
            fur_mapping = {"Short": 1, "Medium": 2, "Long": 3}
            yes_no_not_sure_mapping = {"Yes": 1, "No": 2, "Not Sure": 3}
            health_mapping = {"Healthy": 1, "Minor Injury": 2, "Serious Injury": 3}

            pet_info = {
                "Age": age,
                "Quantity": quantity,
                "Fee": fee,
                "VideoAmt": video_amt,
                "PhotoAmt": photo_amt,
                "Type": type,
                "Gender": gender,
                "MaturitySize": maturity_mapping.get(maturity_size, np.nan),
                "FurLength": fur_mapping.get(fur_length, np.nan),
                "Vaccinated": yes_no_not_sure_mapping.get(vaccinated, np.nan),
                "Dewormed": yes_no_not_sure_mapping.get(dewormed, np.nan),
                "Sterilized": yes_no_not_sure_mapping.get(sterilized, np.nan),
                "Health": health_mapping.get(health, np.nan),
                "StateName": state_name,
                "Breed1Type": breed1_type,
                "Breed2Type": breed2_type,
                "MainBreed": main_breed or np.nan,
                "SecondBreed": second_breed or np.nan,
                "ColorName1": color_name1 or np.nan,
                "ColorName2": color_name2 or np.nan,
                "ColorName3": color_name3 or np.nan,
                "HasName": 1 if pet_name.strip() else 0,
                "IsMixBreed": 1 if second_breed else 0,
                "DescriptionLength": len(description)
            }
            prediction, suggestions = predict_adoption_speed(pet_info, description, pipeline_rf, saved_columns)

            if prediction is not None:
                adoption_labels = {
                    0: ("Estimated to be adopted Same Day", "🟢"),
                    1: ("Estimated to be adopted Within 1 Week", "🟢"),
                    2: ("Estimated to be adopted Within 1 Month", "🟡"),
                    3: ("Estimated to be adopted Within 2-3 Months", "🟡"),
                    4: ("Estimated to have no adoption after 100 Days", "🔴")
                }

                label, emoji = adoption_labels.get(prediction, ("Unknown", "❓"))
                st.success(f"{emoji} **Predicted Adoption Speed:** {label}")

                if suggestions:
                    st.warning("Here are some suggestions to improve the pet's description:")
                    for suggestion in suggestions:
                        st.markdown(f"- {suggestion}")
            else:
                st.error("Prediction failed. Please check the logs for more details.")
