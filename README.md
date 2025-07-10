# CAPTCHA Bot Detection System

This project simulates pointer movement data to distinguish human users from bots using behavioral biometrics and machine learning. The model is trained on synthetic sessions of cursor movements that mimic different types of human users (e.g., power users, elderly users) and bots (e.g., linear bots, ML-mimicking bots).

---

## 🔍 What It Does

- **Generates** 100 realistic pointer movement sessions (50 human, 50 bot)
- **Extracts** behavioral features (velocity stats, path efficiency, entropy)
- **Trains** models (Random Forest and XGBoost) using GridSearchCV
- **Evaluates** performance using ROC-AUC, average precision, CV scores
- **Visualizes** metrics: ROC, PR, confusion matrix, feature importance
- **Predicts** new session labels with confidence scores

---

## ⚙️ Features

- Human user types: casual user, power user, elderly user, CAD designer
- Bot types: simple scripted bots, ML-inspired mimics
- Feature set includes:
  - Velocity (mean, std, max)
  - Time intervals (mean, std)
  - Path efficiency (start-to-end vs. traveled distance)
  - Directional entropy
  - Acceleration variance
  - One-hot encoded time of day (morning, afternoon, evening, night)

---

## 🧠 Models Used

- `RandomForestClassifier` (with hyperparameter tuning)
- `XGBoostClassifier` (boosted decision trees)

---

## 📊 Output

- ROC & PR curves
- Cross-validation bars
- Confusion matrix heatmap
- Feature importance bar chart
- Model comparison summary

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/captcha-bot-detector.git
cd captcha-bot-detector

# Install dependencies (use a virtual environment if needed)
pip install -r requirements.txt

# Run the main script
python main.py
