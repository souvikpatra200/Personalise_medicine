import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Dataset not found: {path}')

    df = pd.read_csv(path, header=0)
    # Drop the first column which is the index
    df = df.iloc[:, 1:]
    required_columns = ['Disease', 'Symptoms', 'Medicine Name']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    df = df[required_columns].dropna(subset=['Disease', 'Symptoms'])
    df['Disease'] = df['Disease'].astype(str).str.strip()
    df['Symptoms'] = df['Symptoms'].astype(str).str.strip()
    df = df[df['Disease'] != '']
    return df


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(base_dir, 'Dataset', 'Medicine_ds1.csv')
    model_dir = os.path.join(base_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    df = load_dataset(dataset_path)
    X = df['Symptoms'].astype(str).tolist()
    y = df['Disease'].astype(str).tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    X_transformed = vectorizer.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if len(df) < 3:
        raise ValueError('Dataset must contain at least 3 rows to train a model.')

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_encoded, test_size=min(0.3, max(0.1, 1.0 / len(df))), random_state=42
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Trained model on {len(df)} entries; test accuracy: {accuracy:.4f}')
    print('Classification report:')
    if len(set(y_test)) == len(label_encoder.classes_):
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    else:
        print(classification_report(y_test, y_pred, labels=sorted(set(y_test)), target_names=label_encoder.inverse_transform(sorted(set(y_test))), zero_division=0))

    with open(os.path.join(model_dir, 'medicine_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(model_dir, 'medicine_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(model_dir, 'medicine_label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    print('Saved model, vectorizer, and label encoder to the model directory.')
