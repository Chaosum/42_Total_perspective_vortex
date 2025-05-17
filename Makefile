# Nom du dossier du venv
VENV_NAME := ./src/venv

# Commande Python
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip

# Pour Windows (si nécessaire)
# PYTHON := $(VENV_NAME)\Scripts\python.exe
# PIP := $(VENV_NAME)\Scripts\pip.exe

# Cible par défaut
setup:
	@if [ ! -d "$(VENV_NAME)" ]; then \
		echo "Création du venv..."; \
		python -m venv $(VENV_NAME); \
	else \
		echo "✅ venv déjà présent."; \
	fi
	@echo "Vérification des dépendances..."
	@. $(VENV_NAME)/bin/activate && \
	tmpfile=$$(mktemp) && \
	pip freeze | sort > $$tmpfile && \
	sort requirements.txt > $${tmpfile}.req && \
	if ! diff -q $$tmpfile $${tmpfile}.req >/dev/null; then \
		echo "📦 Installation des dépendances (différences détectées)..."; \
		pip install -r requirements.txt; \
	else \
		echo "✅ Tous les paquets sont à jour."; \
	fi; \
	rm -f $$tmpfile $${tmpfile}.req


# Pour ouvrir un shell dans le venv
shell:
	@. $(VENV_NAME)/bin/activate && bash

# Pour supprimer l'environnement (clean complet)
clean:
	rm -rf $(VENV_NAME)

.PHONY: setup