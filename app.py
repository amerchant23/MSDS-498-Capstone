import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib

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

# Utility functions
def select_description(x):
    return x['Description']

def clean_description(text):
    text = re.sub(r"<.*?>", "", str(text))  # Remove HTML
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)
    return text

def predict_adoption_speed(pet_info: dict, description: str, pipeline, training_columns: list) -> int:
    input_df = pd.DataFrame([pet_info])
    input_df["Description"] = clean_description(description)

    for col in training_columns:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[training_columns]

    categorical_features = ['Type', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated',
                            'Dewormed', 'Sterilized', 'Health', 'StateName', 'Breed1Type',
                            'Breed2Type', 'MainBreed', 'SecondBreed', 'ColorName1',
                            'ColorName2', 'ColorName3']
    for col in categorical_features:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)

    return pipeline.predict(input_df)[0]

# Load model and labels
try:
    pipeline_rf, saved_columns = joblib.load('pipeline_rf.pkl')
except:
    st.error("Model file not found.")
    st.stop()

try:
    breed_labels_df = pd.read_csv("breed_labels.csv")
    color_labels_df = pd.read_csv("color_labels.csv")
    state_labels_df = pd.read_csv("state_labels.csv")
except Exception as e:
    st.error(f"Error loading label files: {e}")
    st.stop()

# --- App UI ---
st.set_page_config(page_title="🐾 Pet Adoption Predictor", layout="wide")

st.title("🐶 Pet Adoption Speed Predictor")
st.write("Predict how quickly a pet may be adopted based on its characteristics and description.")

with st.sidebar:
    st.header("📋 Pet Profile")

    st.subheader("🐾 Basic Info")
    age = st.number_input("Age (months)", 0, 240)
    quantity = st.number_input("Quantity", 1, 20)
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
            health_mapping = {"Healthy": 1, "Minor Injury": 2, "Serious Injury": 3} # You might want to add "Not Specified": 0

            pet_info = {
                "Age": age,
                "Quantity": quantity,
                "Fee": fee,
                "VideoAmt": video_amt,
                "PhotoAmt": photo_amt,
                "Type": type,
                "Gender": gender,
                "MaturitySize": maturity_mapping.get(maturity_size, np.nan), # Handle potential missing keys
                "FurLength": fur_mapping.get(fur_length, np.nan),
                "Vaccinated": yes_no_not_sure_mapping.get(vaccinated, np.nan),
                "Dewormed": yes_no_not_sure_mapping.get(dewormed, np.nan),
                "Sterilized": yes_no_not_sure_mapping.get(sterilized, np.nan),
                "Health": health_mapping.get(health, np.nan), # Consider adding "Not Specified" to dropdown and mapping
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

            prediction = predict_adoption_speed(pet_info, description, pipeline_rf, saved_columns)

            adoption_labels = {
                0: ("Estimated to be adopted Same Day", "🟢"),
                1: ("Estimated to be adopted Within 1 Week", "🟢"),
                2: ("Estimated to be adopted Within 1 Month", "🟡"),
                3: ("Estimated to be adopted Within 2-3 Months", "🟡"),
                4: ("Estimated to have no adoption after 100 Days", "🔴")
            }

            label, emoji = adoption_labels.get(prediction, ("Unknown", "❓"))
            st.success(f"{emoji} **Predicted Adoption Speed:** {label}")
