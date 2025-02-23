# Makefile

# Variables
TRAIN_DATA = churn-bigml-80.csv
TEST_DATA = churn-bigml-20.csv
MODEL_FILE = model.pkl

# Installer les dépendances
install:
	venv/bin/pip install -r requirements.txt

# Vérifier la qualité du code
check:
	black . && flake8 . && bandit -r . && mypy --ignore-missing-imports .

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
