# 🚀 ProjetFinS1 — Prédiction du Risque Cardiovasculaire

## 📌 Description

Ce projet permet de prédire le risque de maladie cardiovasculaire à partir de données de santé (âge, tension, BMI, etc.) grâce à un modèle de machine learning, puis d’afficher le résultat via une application web Flask.

---

## ⚙️ Installation

Clone le projet :

```bash
git clone https://github.com/gh832/projetfins1.git
cd projetfins1
```

Créer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l’environnement :

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## 🧠 Entraîner le modèle

```bash
python scripts/train_model.py
```

➡️ Cela va générer :

* `model.pkl`
* `scaler.pkl`

---

## 🌐 Lancer l’application

```bash
python dashboard/app.py
```

Puis ouvrir le navigateur sur l’adresse affichée (ex: http://127.0.0.1:5000)

---

## 📁 Structure du projet

```text
projetfins1/
├── README.md
├── requirements.txt
├── .gitignore
├── dashboard/
│   ├── app.py
│   ├── templates/
│   └── static/
├── scripts/
│   └── train_model.py
├── data_clean/
├── data_raw/
├── notebooks/
```

---

## 📊 Variables utilisées

* age_years
* BMI
* ap_hi (pression systolique)
* ap_lo (pression diastolique)
* cholesterol
* gluc
* smoke
* alco
* active
* gender




## 👤 Auteur

gh832
