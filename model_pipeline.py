# model_pipeline.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data(train_path, test_path):
    """Charge les fichiers CSV et retourne les dataframes."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def prepare_data(train_data, test_data):
    """Prépare les données pour l'entraînement."""
    data = pd.concat([train_data, test_data], ignore_index=True)

    # Encodage des colonnes catégorielles
    label_encoders = {}
    for col in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Normalisation des caractéristiques
    scaler = StandardScaler()
    features = data.drop(columns=["Churn"], errors="ignore")
    target = data["Churn"]
    features_scaled = scaler.fit_transform(features)

    # Vérifiez la forme de features_scaled
    print(f"Forme de features_scaled : {features_scaled.shape}")

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, label_encoders


def train_model(X_train, y_train):
    """Entraîne un modèle de régression linéaire."""
    # Vérifiez la forme de X_train avant entraînement
    print(f"Forme de X_train avant entraînement : {X_train.shape}")

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle et retourne les métriques."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    return mse, r2


def save_prepared_data(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    label_encoders,
    filename="prepared_data.pkl",
):
    """Sauvegarde les données préparées dans un fichier."""
    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler,
            "label_encoders": label_encoders,
        },
        filename,
    )
    print(f"\n✅ Données préparées sauvegardées dans '{filename}'")


def load_prepared_data(filename="prepared_data.pkl"):
    """Charge les données préparées depuis un fichier."""
    data = joblib.load(filename)
    print(f"\n✅ Données préparées chargées depuis '{filename}'")
    return (
        data["X_train"],
        data["X_test"],
        data["y_train"],
        data["y_test"],
        data["scaler"],
        data["label_encoders"],
    )
