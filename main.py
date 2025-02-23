# main.py
import mlflow
import mlflow.sklearn
import argparse
import joblib
from model_pipeline import (
    load_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_prepared_data,
    load_prepared_data,
)

# Définir un nom d'expérience MLflow
mlflow.set_experiment("medical_ocr_experiment")

# Variables globales
X_train, X_test, y_train, y_test, scaler, label_encoders, model = (
    None,
    None,
    None,
    None,
    None,
    None,
    None,
)


def execute_command(
    command,
    train_path="churn-bigml-80.csv",
    test_path="churn-bigml-20.csv",
    model_path="model.pkl",
    save_path=None,
):
    """Exécute la commande spécifiée (prepare, train, test)."""
    global X_train, X_test, y_train, y_test, scaler, label_encoders, model

    if command == "prepare":
        # Charger et préparer les données
        train_data, test_data = load_data(train_path, test_path)
        X_train, X_test, y_train, y_test, scaler, label_encoders = prepare_data(
            train_data, test_data
        )

        # Sauvegarder les données préparées
        save_prepared_data(
            X_train,
            X_test,
            y_train,
            y_test,
            scaler,
            label_encoders,
            "prepared_data.pkl",
        )

    elif command == "train":
        try:
            # Charger les données préparées
            X_train, X_test, y_train, y_test, scaler, label_encoders = (
                load_prepared_data("prepared_data.pkl")
            )
        except FileNotFoundError:
            print(
                "\n⚠ Fichier 'prepared_data.pkl' introuvable. Préparez d'abord les données avec la commande 'prepare'"
            )
            return

        # Entraîner le modèle avec MLflow
        print("\n# Entraînement du modèle avec MLflow...")
        with mlflow.start_run():
            model = train_model(X_train, y_train)

            # Log des hyperparamètres
            mlflow.log_param("algorithm", "Linear Regression")
            mlflow.log_param("train_size", len(X_train))

            # Log du modèle entraîné
            mlflow.sklearn.log_model(model, "model")
            print("\n✅ Modèle entraîné et enregistré sur MLflow")

            # Sauvegarder le modèle dans un fichier
            joblib.dump(model, model_path)
            print(f"\n✅ Modèle sauvegardé dans '{model_path}'")

    elif command == "test":
        try:
            # Charger les données préparées
            X_train, X_test, y_train, y_test, scaler, label_encoders = (
                load_prepared_data("prepared_data.pkl")
            )
        except FileNotFoundError:
            print(
                "\n⚠ Fichier 'prepared_data.pkl' introuvable. Préparez d'abord les données avec la commande 'prepare'"
            )
            return

        try:
            # Charger le modèle entraîné
            model = joblib.load(model_path)
            print(f"\n✅ Modèle chargé depuis '{model_path}'")
        except FileNotFoundError:
            print(
                f"\n⚠ Fichier '{model_path}' introuvable. Entraînez d'abord le modèle avec la commande 'train'"
            )
            return

        # Évaluer le modèle
        print("\n# Évaluation du modèle avec MLflow...")
        with mlflow.start_run():
            mse, r2 = evaluate_model(model, X_test, y_test)

            # Log des métriques de performance
            mlflow.log_metric("Mean Squared Error", mse)
            mlflow.log_metric("R² Score", r2)

            print(f"\n✅ Mean Squared Error: {mse}")
            print(f"✅ R² Score: {r2}")
            print("\n✅ Métriques enregistrées sur MLflow")


# Point d'entrée du script
if __name__ == "__main__":
    # Interface en ligne de commande
    parser = argparse.ArgumentParser(
        description="Exécuter des commandes pour entraîner et évaluer un modèle."
    )
    parser.add_argument(
        "command", choices=["prepare", "train", "test"], help="Commande à exécuter."
    )
    parser.add_argument(
        "--train_path",
        default="churn-bigml-80.csv",
        help="Chemin vers le fichier d'entraînement.",
    )
    parser.add_argument(
        "--test_path",
        default="churn-bigml-20.csv",
        help="Chemin vers le fichier de test.",
    )
    parser.add_argument(
        "--model_path",
        default="model.pkl",
        help="Chemin pour charger/sauvegarder le modèle.",
    )
    parser.add_argument("--save_path", help="Chemin pour sauvegarder les résultats.")

    args = parser.parse_args()
    execute_command(
        args.command, args.train_path, args.test_path, args.model_path, args.save_path
    )
