# Makefile

# Variables
TRAIN_DATA = churn-bigml-80.csv
TEST_DATA = churn-bigml-20.csv
MODEL_FILE = model.pkl
PYTHON_FILES = main.py model_pipeline.py  # Specify the files to be checked

# Installer les dépendances
install:
	venv/bin/pip install -r requirements.txt

# Vérifier la qualité du code uniquement pour main.py et model_pipeline.py
check:
	black $(PYTHON_FILES) && pylint $(PYTHON_FILES) && bandit -r $(PYTHON_FILES) && mypy --ignore-missing-imports $(PYTHON_FILES)

# Préparer les données
prepare:
	python main.py prepare --train_path $(TRAIN_DATA) --test_path $(TEST_DATA)

# Entraîner le modèle
train:
	python main.py train --train_path $(TRAIN_DATA) --test_path $(TEST_DATA) --model_path $(MODEL_FILE)

# Tester le modèle
test:
	python main.py test --train_path $(TRAIN_DATA) --test_path $(TEST_DATA) --model_path $(MODEL_FILE)

# Nettoyer les fichiers temporaires
clean:
	rm -f prepared_data.pkl $(MODEL_FILE)

# Exécuter toutes les étapes en une seule commande
all: install check prepare train test

