import sklearn
import numpy as np
import pandas as pd
import pickle
import re
import difflib
import os
from flask import Flask, request, render_template, jsonify
from collections import defaultdict

# flask app
app = Flask(__name__)
base_dir = os.path.dirname(os.path.realpath(__file__))

# Load the new disease/symptoms/medicine dataset and use it exclusively.
dataset_path = os.path.join(base_dir, 'Dataset', 'Medicine_ds1.csv')
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f'Primary dataset not found: {dataset_path}')

raw_dataset = pd.read_csv(dataset_path, header=0)
# Drop the first column which is the index
raw_dataset = raw_dataset.iloc[:, 1:]
required_columns = ['Disease', 'Symptoms', 'Medicine Name']
for col in required_columns:
    if col not in raw_dataset.columns:
        raise ValueError(f'Missing required column in dataset: {col}')

raw_dataset = raw_dataset[required_columns].dropna(subset=['Disease', 'Symptoms'])
raw_dataset['Disease'] = raw_dataset['Disease'].astype(str).str.strip()
raw_dataset['Symptoms'] = raw_dataset['Symptoms'].astype(str).str.strip()
raw_dataset['Medicine Name'] = raw_dataset['Medicine Name'].astype(str).str.strip()
raw_dataset = raw_dataset[raw_dataset['Disease'] != '']

# Build lookup dictionaries from the new dataset.
disease_to_symptoms = {}
disease_to_medicines = defaultdict(list)
disease_labels = {}
for _, row in raw_dataset.iterrows():
    disease = str(row['Disease']).strip()
    disease_lower = disease.lower()
    disease_to_symptoms[disease_lower] = str(row['Symptoms']).strip()
    disease_labels[disease_lower] = disease
    medicines = re.split(r',|;| or ', str(row['Medicine Name']))
    for medicine in medicines:
        medicine = medicine.strip()
        if medicine:
            disease_to_medicines[disease_lower].append(medicine)

# Load a trained model for the new dataset if available.
model_path = os.path.join(base_dir, 'model', 'medicine_model.pkl')
vectorizer_path = os.path.join(base_dir, 'model', 'medicine_vectorizer.pkl')
label_encoder_path = os.path.join(base_dir, 'model', 'medicine_label_encoder.pkl')

svc = None
vectorizer = None
label_encoder = None
if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(label_encoder_path):
    with open(model_path, 'rb') as f:
        svc = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

# Optional medicine details enrichment (if available).
medicine_details = None
medicine_details_path_csv = os.path.join(base_dir, 'Dataset', 'Medicine_Details.csv')
medicine_details_path_xlsx = os.path.join(base_dir, 'Dataset', 'Medicine_Details.xlsx')
if os.path.exists(medicine_details_path_csv):
    medicine_details = pd.read_csv(medicine_details_path_csv)
elif os.path.exists(medicine_details_path_xlsx):
    medicine_details = pd.read_excel(medicine_details_path_xlsx, sheet_name='Medicine_Details')

if medicine_details is not None:
    medicine_details.columns = [c.strip() for c in medicine_details.columns]
    medicine_details.fillna('', inplace=True)

# Basic medicines information (safe, commonly used options).
# Only general guidance is provided — not a substitute for medical advice.

# Basic medicines information (safe, commonly used options).
# Only general guidance is provided — not a substitute for medical advice.
MEDICINES_INFO = {
    'Paracetamol': {
        'purpose': 'Relieves pain and reduces fever',
        'dosage': '500–1000 mg every 4–6 hours as needed (max ~3000 mg/day for adults)',
        'precautions': 'Avoid exceeding recommended dose; check liver disease before use; avoid with other acetaminophen-containing products.',
        'when_to_consult': 'If fever persists >3 days, severe pain, signs of liver problems (jaundice), or overdose.'
    },
    'Ibuprofen': {
        'purpose': 'Nonsteroidal anti-inflammatory for pain, inflammation, and fever',
        'dosage': '200–400 mg every 4–6 hours as needed (typical OTC max 1200 mg/day)',
        'precautions': 'Take with food to reduce stomach upset; avoid if history of peptic ulcer, uncontrolled high blood pressure, or severe kidney disease.',
        'when_to_consult': 'If severe stomach pain, blood in stools, shortness of breath, or symptoms persist.'
    },
    'Cetirizine': {
        'purpose': 'Oral antihistamine for allergic symptoms (sneezing, itching, runny nose)',
        'dosage': '10 mg once daily (adults)',
        'precautions': 'May cause drowsiness in some people; use caution when driving.',
        'when_to_consult': 'If symptoms persist despite treatment or if severe allergic reaction occurs.'
    },
    'Loratadine': {
        'purpose': 'Non-drowsy oral antihistamine for allergy symptoms',
        'dosage': '10 mg once daily (adults)',
        'precautions': 'Generally well tolerated; check interactions with other medications.',
        'when_to_consult': 'If no improvement or symptoms worsen.'
    },
    'Amoxicillin': {
        'purpose': 'Broad-spectrum antibiotic used for some bacterial infections (prescription only)',
        'dosage': 'Typical adult dose varies by infection (commonly 500 mg every 8 hours); follow prescriber instructions',
        'precautions': 'Only use when prescribed by a clinician; report penicillin allergy or rash.',
        'when_to_consult': 'If signs of allergic reaction (rash, swelling, breathing problems), severe diarrhea, or no improvement.'
    },
    'Azithromycin': {
        'purpose': 'Antibiotic used for certain respiratory and other bacterial infections (prescription only)',
        'dosage': 'Follow prescriber instructions; common short-course regimens exist',
        'precautions': 'Prescription-only; report liver disease or heart rhythm issues.',
        'when_to_consult': 'If allergic reaction, severe diarrhea, or symptoms do not improve.'
    },
    'Topical hydrocortisone': {
        'purpose': 'Mild topical steroid for itching and inflammation (skin)',
        'dosage': 'Apply a thin layer to affected area 1–2 times daily as directed',
        'precautions': 'Avoid long-term use on large areas; do not use on infected skin without advice.',
        'when_to_consult': 'If skin worsens, shows signs of infection, or no improvement.'
    }
}
# Map general category names (as used in dataset) to example medicines
CATEGORY_TO_EXAMPLES = {
    'Antiviral drugs': ['Acyclovir'],
    'Pain relievers': ['Paracetamol', 'Ibuprofen'],
    'Antihistamines': ['Cetirizine', 'Loratadine'],
    'Antipyretics': ['Paracetamol', 'Ibuprofen'],
    'Topical antifungal': ['Clotrimazole', 'Ketoconazole'],
    'Antifungal Cream': ['Clotrimazole'],
    'Antibiotics': ['Amoxicillin', 'Azithromycin'],
    'Oral antibiotics': ['Amoxicillin', 'Azithromycin'],
    'Topical antibiotics': ['Amoxicillin'],
    'Antiemetic drugs': ['Ondansetron'],
    'Antidiarrheal drugs': ['Loperamide'],
    'Anticholinergics': ['Ipratropium'],
    'Anticonvulsants': ['Carbamazepine'],
    'Antihypertensive medications': ['Amlodipine', 'Metoprolol'],
    'Diuretics': ['Furosemide'],
    'Beta-blockers': ['Metoprolol'],
    'ACE inhibitors': ['Enalapril'],
    'Calcium channel blockers': ['Amlodipine'],
    'Antimalarial drugs': ['Chloroquine'],
    'Antiviral medications': ['Acyclovir'],
    'Antiretroviral drugs': ['Lamivudine'],
    'Bronchodilators': ['Salbutamol'],
    'Inhaled corticosteroids': ['Budesonide'],
    'Corticosteroids': ['Prednisone'],
    'NSAIDs': ['Ibuprofen'],
    'Proton Pump Inhibitors (PPIs)': ['Omeprazole'],
    'H2 Blockers': ['Ranitidine'],
    'Insulin': ['Insulin'],
    'Metformin': ['Metformin'],
    'Sulfonylureas': ['Glipizide'],
    'DPP-4 inhibitors': ['Sitagliptin'],
    'GLP-1 receptor agonists': ['Liraglutide'],
    'Analgesics': ['Paracetamol', 'Ibuprofen'],
    'Decongestants': ['Pseudoephedrine'],
    'Antacids': ['Ranitidine'],
    'IV fluids': [],
    'Thrombolytic drugs': ['Alteplase', 'Streptokinase'],
    'Blood thinners': ['Warfarin', 'Heparin'],
    'Clot-dissolving medications': ['Alteplase', 'Streptokinase'],
    'Medications for itching': ['Hydrocortisone', 'Calamine lotion'],
    'Topical treatments': ['Hydrocortisone', 'Clotrimazole'],
    'Anticoagulants': ['Warfarin', 'Heparin'],
    'Antifungal drugs': ['Clotrimazole', 'Ketoconazole'],
    'Antihypertensive': ['Amlodipine', 'Metoprolol'],
    'Antiviral': ['Acyclovir'],
    'Antidiarrheal': ['Loperamide'],
    'Antipyretic': ['Paracetamol', 'Ibuprofen'],
    'Antiemetic': ['Ondansetron'],
    'Pain relievers': ['Paracetamol', 'Ibuprofen'],
    'Topical treatments': ['Hydrocortisone', 'Clotrimazole'],
}

SYMPTOM_TO_MEDICINE = defaultdict(list)

GENERIC_MEDICATION_KEYWORDS = {
    'drugs', 'medications', 'treatments', 'therapy', 'analgesics', 'antibiotics',
    'antihistamines', 'antipyretics', 'antidiarrheal', 'antiviral', 'antimalarial',
    'anticoagulants', 'beta-blockers', 'calcium channel blockers', 'ace inhibitors',
    'diuretics', 'antacids', 'antispasmodics', 'corticosteroids', 'topical treatments',
    'oral medications', 'antifungal', 'vaccination', 'antihypertensive', 'pain relievers',
    'clot-dissolving', 'thrombolytic', 'blood thinners'
}

def is_generic_medication_entry(name):
    if not name:
        return False
    text = str(name).lower()
    return any(keyword in text for keyword in GENERIC_MEDICATION_KEYWORDS)

# Normalize medicine names for lookup.
def normalize_medicine_name(name):
    if not isinstance(name, str):
        return ""
    cleaned = name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

# Lookup medicine information in the Medicine_Details spreadsheet.
def get_medicine_detail_from_excel(name):
    normalized = normalize_medicine_name(name)
    if not normalized or medicine_details is None:
        return None

    # Try exact matches on the product name first.
    exact_matches = medicine_details[medicine_details['Medicine Name'].str.lower().str.strip() == normalized]
    if not exact_matches.empty:
        row = exact_matches.iloc[0]
    else:
        # Try matching on composition and uses fields.
        has_name = medicine_details['Medicine Name'].str.lower().str.contains(re.escape(normalized), na=False)
        has_comp = medicine_details['Composition'].str.lower().str.contains(re.escape(normalized), na=False)
        has_uses = medicine_details['Uses'].str.lower().str.contains(re.escape(normalized), na=False)
        matches = medicine_details[has_name | has_comp | has_uses]
        if matches.empty:
            return None
        row = matches.iloc[0]

    return {
        'name': row.get('Medicine Name', name),
        'purpose': row.get('Uses', ''),
        'dosage': row.get('Composition', ''),
        'precautions': row.get('Side_effects', ''),
        'when_to_consult': 'Consult a qualified healthcare provider before prescribing.'
    }

#============================================================
# custome and helping functions
#==========================helper funtions================

def helper(dis):
    disease_key = str(dis).strip().lower()
    if not disease_key:
        return "", [], [], [], []

    med = []
    medicine_names = disease_to_medicines.get(disease_key, [])
    for name in medicine_names:
        info = MEDICINES_INFO.get(name)
        if info:
            med.append({
                'name': name,
                'purpose': info.get('purpose', ''),
                'dosage': info.get('dosage', ''),
                'precautions': info.get('precautions', ''),
                'when_to_consult': info.get('when_to_consult', '')
            })
            continue

        detail_info = get_medicine_detail_from_excel(name)
        if detail_info:
            med.append(detail_info)
            continue

        med.append({
            'name': name,
            'purpose': '',
            'dosage': '',
            'precautions': 'Follow prescriber or pharmacist advice.',
            'when_to_consult': 'Consult a healthcare professional for specific guidance.'
        })

    return "", [], med, [], []


# disease_labels is already created from the dataset to preserve original disease names.

def get_predicted_value(patient_symptoms):
    if not patient_symptoms:
        return "Unknown Disease"

    input_text = " ".join([str(sym).strip() for sym in patient_symptoms if str(sym).strip()])
    if not input_text:
        return "Unknown Disease"

    if svc is not None and vectorizer is not None and label_encoder is not None:
        try:
            X_pred = vectorizer.transform([input_text])
            predicted_code = svc.predict(X_pred)[0]
            predicted_disease = label_encoder.inverse_transform([predicted_code])[0]
            if predicted_disease:
                return predicted_disease
        except Exception:
            pass

    best_disease = None
    best_score = 0.0
    normalized_input = input_text.lower()
    for disease, symptom_text in disease_to_symptoms.items():
        score = difflib.SequenceMatcher(None, normalized_input, symptom_text.lower()).ratio()
        if score > best_score:
            best_score = score
            best_disease = disease
    if best_disease and best_score >= 0.2:
        return disease_labels.get(best_disease, best_disease)

    return "Unknown Disease"


def get_medication_suggestions_for_symptoms(symptoms):
    suggestions = []
    seen = set()
    user_text = " ".join([str(sym).strip().lower() for sym in symptoms if str(sym).strip()])
    if not user_text:
        return suggestions

    for disease, symptom_text in disease_to_symptoms.items():
        if any(token in symptom_text.lower() for token in user_text.split() if len(token) > 3):
            for name in disease_to_medicines.get(disease, []):
                if name in seen:
                    continue
                seen.add(name)
                info = MEDICINES_INFO.get(name)
                if info:
                    suggestions.append({
                        'name': name,
                        'purpose': info.get('purpose', ''),
                        'dosage': info.get('dosage', ''),
                        'precautions': info.get('precautions', ''),
                        'when_to_consult': info.get('when_to_consult', '')
                    })
                else:
                    detail_info = get_medicine_detail_from_excel(name)
                    if detail_info:
                        suggestions.append(detail_info)
                    else:
                        suggestions.append({
                            'name': name,
                            'purpose': '',
                            'dosage': '',
                            'precautions': 'Follow prescriber or pharmacist advice.',
                            'when_to_consult': 'Consult a healthcare professional for specific guidance.'
                        })
    return suggestions









# # Initialize the TextBlob object for spelling correction
def correct_spelling(symptom):
    # Correct the spelling of a single symptom
    blob = TextBlob(symptom)
    return str(blob.correct())

# Try a fuzzy match against known symptoms and synonym phrases.
def find_best_symptom_match(item, cutoff=0.75):
    item = str(item).strip().lower()
    if not item:
        return None

    best_match = None
    best_score = 0.0
    for key, synonyms in symptom_mapping.items():
        score = difflib.SequenceMatcher(None, item, key).ratio()
        if score > best_score:
            best_score = score
            best_match = key
        for syn in synonyms:
            score = difflib.SequenceMatcher(None, item, syn).ratio()
            if score > best_score:
                best_score = score
                best_match = key

    return best_match if best_score >= cutoff else None

# Normalize input symptom text to keys used by the model and mappings.
def normalize_symptom(symptom):
    item = str(symptom).strip().lower()
    item = item.replace('-', ' ').replace('_', ' ').replace('.', ' ').strip()
    item = ' '.join(item.split())
    if not item:
        return None
    if item in symptoms_dict:
        return item
    for key, synonyms in symptom_mapping.items():
        if item == key or item in synonyms:
            return key
    fuzzy = find_best_symptom_match(item)
    if fuzzy:
        return fuzzy
    return item

# Try to resolve symptom by normalization, mapping, or spelling correction.
def resolve_symptom(symptom):
    normalized = normalize_symptom(symptom)
    if not normalized:
        return None
    if normalized in symptoms_dict:
        return normalized
    for key, synonyms in symptom_mapping.items():
        if normalized == key or normalized in synonyms:
            return key
    corrected = normalize_symptom(correct_spelling(normalized))
    if corrected and corrected in symptoms_dict:
        return corrected
    for key, synonyms in symptom_mapping.items():
        if corrected == key or corrected in synonyms:
            return key
    return None

symptom_mapping = defaultdict(lambda: "unknown", {
    "itching": ["itching", "itchy", "itchy skin", "skin itching"],
    "skin_rash": ["skin rash", "rash", "dermatitis", "rashes", "erythema"],
    "nodal_skin_eruptions": ["nodal skin eruptions", "skin eruptions", "bumps"],
    "continuous_sneezing": ["continuous sneezing", "sneezing", "allergic sneezing"],
    "shivering": ["shivering", "trembling", "shaky"],
    "chills": ["chills", "cold sensation", "cold", "feeling cold"],
    "joint_pain": ["joint pain", "arthralgia", "aching joints", "joint ache", "pain in joints"],
    "stomach_pain": ["stomach pain", "abdominal pain", "belly ache", "stomach ache", "tummy pain"],
    "acidity": ["acidity", "heartburn", "acid reflux"],
    "ulcers_on_tongue": ["ulcers on tongue", "tongue ulcers", "mouth sores"],
    "muscle_wasting": ["muscle wasting", "muscle loss", "muscle atrophy"],
    "vomiting": ["vomiting", "emesis", "throwing up", "throwing up"],
    "burning_micturition": ["burning micturition", "burning urination", "painful urination"],
    "spotting_urination": ["spotting urination", "blood in urine", "hematuria"],
    "fatigue": ["fatigue", "tiredness", "exhaustion", "weariness", "low energy"],
    "weight_gain": ["weight gain", "increased weight"],
    "anxiety": ["anxiety", "nervousness", "worry"],
    "cold_hands_and_feets": ["cold hands and feet", "cold extremities"],
    "mood_swings": ["mood swings", "emotional changes"],
    "weight_loss": ["weight loss", "decreased weight"],
    "restlessness": ["restlessness", "agitation"],
    "lethargy": ["lethargy", "sluggishness"],
    "patches_in_throat": ["patches in throat", "throat patches", "throat lesions"],
    "irregular_sugar_level": ["irregular sugar level", "unstable glucose", "blood sugar fluctuations"],
    "cough": ["cough", "coughing", "dry cough", "wet cough", "coughing a lot"],
    "high_fever": ["high fever", "elevated temperature", "temperature"],
    "sunken_eyes": ["sunken eyes", "hollow eyes"],
    "breathlessness": ["breathlessness", "shortness of breath", "dyspnea", "difficulty breathing", "can\'t breathe"],
    "sweating": ["sweating", "perspiration"],
    "dehydration": ["dehydration", "fluid loss"],
    "indigestion": ["indigestion", "upset stomach"],
    "headache": ["headache", "head pain", "migraine"],
    "yellowish_skin": ["yellowish skin", "jaundice"],
    "dark_urine": ["dark urine"],
    "nausea": ["nausea", "queasiness"],
    "loss_of_appetite": ["loss of appetite", "no appetite", "anorexia"],
    "pain_behind_the_eyes": ["pain behind the eyes", "eye pain"],
    "back_pain": ["back pain", "lower back pain"],
    "constipation": ["constipation", "difficulty passing stool"],
    "abdominal_pain": ["abdominal pain", "belly pain", "stomach ache"],
    "diarrhoea": ["diarrhoea", "loose stools"],
    "mild_fever": ["mild fever", "fever", "low-grade fever"],
    "yellow_urine": ["yellow urine"],
    "yellowing_of_eyes": ["yellowing of eyes", "scleral icterus"],
    "acute_liver_failure": ["acute liver failure", "hepatic failure"],
    "fluid_overload": ["fluid overload", "edema"],
    "swelling_of_stomach": ["swelling of stomach", "abdominal bloating"],
    "swelled_lymph_nodes": ["swelled lymph nodes", "enlarged lymph nodes"],
    "malaise": ["malaise", "general discomfort"],
    "blurred_and_distorted_vision": ["blurred and distorted vision", "blurry vision"],
    "phlegm": ["phlegm", "mucus"],
    "throat_irritation": ["throat irritation", "sore throat", "throat pain"],
    "redness_of_eyes": ["redness of eyes", "bloodshot eyes", "red eyes"],
    "sinus_pressure": ["sinus pressure", "sinus congestion"],
    "runny_nose": ["runny nose", "rhinorrhea", "nasal drip"],
    "congestion": ["congestion", "nasal blockage", "stuffy nose", "blocked nose"],
    "chest_pain": ["chest pain", "angina"],
    "weakness_in_limbs": ["weakness in limbs", "limb weakness"],
    "fast_heart_rate": ["fast heart rate", "tachycardia"],
    "pain_during_bowel_movements": ["pain during bowel movements", "painful defecation"],
    "pain_in_anal_region": ["pain in anal region", "anal pain"],
    "bloody_stool": ["bloody stool", "rectal bleeding"],
    "irritation_in_anus": ["irritation in anus", "anal itching"],
    "neck_pain": ["neck pain", "cervical pain"],
    "dizziness": ["dizziness", "lightheadedness", "dizzy", "vertigo", "light headed"],
    "cramps": ["cramps", "muscle cramps", "spasms"],
    "bruising": ["bruising", "hematoma"],
    "obesity": ["obesity", "overweight"],
    "swollen_legs": ["swollen legs", "leg edema"],
    "swollen_blood_vessels": ["swollen blood vessels", "varicose veins"],
    "puffy_face_and_eyes": ["puffy face and eyes", "facial swelling"],
    "enlarged_thyroid": ["enlarged thyroid", "goiter"],
    "brittle_nails": ["brittle nails", "weak nails"],
    "swollen_extremeties": ["swollen extremities", "swollen arms and legs"],
    "excessive_hunger": ["excessive hunger", "polyphagia"],
    "extra_marital_contacts": ["extra marital contacts", "multiple sexual partners"],
    "drying_and_tingling_lips": ["drying and tingling lips", "lip dryness"],
    "slurred_speech": ["slurred speech", "dysarthria"],
    "knee_pain": ["knee pain", "pain in the knees"],
    "hip_joint_pain": ["hip joint pain", "hip pain"],
    "muscle_weakness": ["muscle weakness", "muscle fatigue"],
    "stiff_neck": ["stiff neck", "neck stiffness"],
    "swelling_joints": ["swelling joints", "joint swelling"],
    "movement_stiffness": ["movement stiffness", "rigidity"],
    "spinning_movements": ["spinning movements", "vertigo"],
    "loss_of_balance": ["loss of balance", "balance problems"],
    "unsteadiness": ["unsteadiness", "lack of balance"],
    "weakness_of_one_body_side": ["weakness of one body side", "hemiparesis"],
    "loss_of_smell": ["loss of smell", "anosmia"],
    "bladder_discomfort": ["bladder discomfort", "bladder pain"],
    "foul_smell_of_urine": ["foul smell of urine", "smelly urine"],
    "continuous_feel_of_urine": ["continuous feel of urine", "urgency to urinate"],
    "passage_of_gases": ["passage of gases", "flatulence"],
    "internal_itching": ["internal itching"],
    "toxic_look_(typhos)": ["toxic look (typhos)", "septic appearance"],
    "depression": ["depression", "low mood"],
    "irritability": ["irritability", "easily annoyed"],
    "muscle_pain": ["muscle pain", "myalgia", "body ache", "muscle ache", "body pain"],
    "altered_sensorium": ["altered sensorium", "confusion"],
    "red_spots_over_body": ["red spots over body", "rash with red spots"],
    "belly_pain": ["belly pain", "abdominal pain", "tummy pain"],
    "abnormal_menstruation": ["abnormal menstruation", "irregular periods"],
    "dischromic_patches": ["dischromic patches", "skin discoloration"],
    "watering_from_eyes": ["watering from eyes", "teary eyes"],
    "increased_appetite": ["increased appetite", "hyperphagia"],
    "polyuria": ["polyuria", "excessive urination"],
    "family_history": ["family history", "genetic predisposition"],
    "mucoid_sputum": ["mucoid sputum", "mucus in sputum"],
    "rusty_sputum": ["rusty sputum", "blood-tinged sputum"],
    "lack_of_concentration": ["lack of concentration", "difficulty focusing"],
    "visual_disturbances": ["visual disturbances", "vision problems"],
    "receiving_blood_transfusion": ["receiving blood transfusion"],
    "receiving_unsterile_injections": ["receiving unster"]
    # Add more symptoms and their synonyms here
})


# creating routes========================================


@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        location = request.form.get('location')
        symptoms = request.form.get('symptoms')
        # mysysms = request.form.get('mysysms')
        # print(mysysms)
        print(symptoms)
        if symptoms =="Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        
        if not symptoms:
            message = "Please enter symptoms."
            return render_template('index.html', message=message) 
        
        else:

            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            # Remove any extra characters, if any
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            symptom_based_meds = get_medication_suggestions_for_symptoms(user_symptoms)
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            # If the model fails to find a disease-level medication list, use symptom-level fallback suggestions.
            if not medications:
                medications = symptom_based_meds

            # precautions, medications, diets and workout are lists (may be empty)
            my_precautions = precautions  # already a list

            return render_template(
                'index.html',
                name=name,
                age=age,
                location=location,
                symptoms=symptoms,
                predicted_disease=predicted_disease,
                dis_des=dis_des,
                my_precautions=my_precautions,
                medications=medications,
                my_diet=rec_diet,
                workout=workout
            )

    return render_template('index.html')



# about view funtion and path
@app.route('/about')
def about():
    return render_template("about.html")
# contact view funtion and path
@app.route('/contact')
def contact():
    return render_template("contact.html")

# developer view funtion and path
@app.route('/developer')
def developer():
    return render_template("developer.html")

# about view funtion and path
@app.route('/blog')
def blog():
    return render_template("blog.html")

# about view funtion and path
@app.route('/upload',  methods=['GET', 'POST'])
def upload():
    return render_template("upload.html")

if __name__ == '__main__':

    app.run(debug=True)