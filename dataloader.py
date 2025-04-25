import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_db(db=1, exercise="B", samples_per_class=500, reduce=True) -> list[pd.DataFrame]:
    # Get db directory
    if db == 1:
        if exercise == "B":
            db_file_path = "./data/db1/db1-exB-gestures.csv"
        elif exercise == "C":
            db_file_path = "./data/db1/db1-exC-gestures.csv"
    else:
        if exercise == "B":
            db_file_path = "./data/db2/db2-exB-gestures-ds.csv"
        elif exercise == "C":
            db_file_path = "./data/db2/db2-exC-gestures-ds.csv"

    # Get file as dataframe
    df = pd.read_csv(db_file_path)

    # Drop null values
    df = df.dropna()

    if reduce:
        # Shuffle dataframe
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)

        # Keep only N samples of each class
        new_df = pd.DataFrame()
        for gesture in df['gesture'].unique():
            gesture_df = df[df['gesture'] == gesture].head(samples_per_class)
            new_df = pd.concat([new_df, gesture_df])

        return new_df
    else:
        return df

def data_split_emg(db_df, test_size=0.2, standardize=True):
    emg_features = [col for col in db_df.columns if col.startswith('emg_')]
    X = db_df[emg_features]
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    y = db_df['gesture'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=0)
    return X_train, X_test, y_train, y_test

def has_model_weights(filename="rfc_db1.pkl"):
    # Return whether the given model weights file exists
    return os.path.exists("./data/model_weights/"+filename)

def load_model_weights(filename="rfc_db1.pkl"):
    # Load model weights from file
    with open("./data/model_weights/"+filename, "rb") as f:
        model = pickle.load(f)
    return model

def save_model_weights(model, filename="rfc_db1.pkl"):
    # Save model weights to file
    with open("./data/model_weights/"+filename, "wb") as f:
        pickle.dump(model, f, protocol=5)
    return