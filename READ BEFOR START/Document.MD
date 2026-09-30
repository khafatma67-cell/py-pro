# Student Performance Predictor 🎓

A machine learning project that predicts whether a student will pass, based on study habits, family background and social factors. Includes a full ML pipeline and an interactive Streamlit app.

## Dataset
UCI **Student Performance** (Math): https://archive.ics.uci.edu/dataset/320/student+performance  
Download `student-mat.csv` and place it in `data/student-mat.csv` (separator is `;`).  
Target: `passed = 1` if final grade `G3 >= 10`.  
Note: `G1` and `G2` (earlier grades) are deliberately excluded, so the model works as an *early warning* system.  
If the file is missing, `train.py` falls back to synthetic demo data (do not report those numbers as real results).

## Quick start
```bash
pip install -r requirements.txt
python train.py          # EDA + training + evaluation -> model.joblib, outputs/
streamlit run app.py     # web app
```

## Languages
The app supports **Arabic (RTL), English and French**. Switch with the buttons at the top, or share a direct link such as `https://your-app-url/?lang=fr`.
To add a language, copy one block in `i18n.py` and add it to `LANGS`.

## Use it on any phone, tablet or laptop
The app is a responsive web page, so nothing needs installing on the user's device.

**Option 1: public link (recommended)**, free on Streamlit Community Cloud:
1. Run `python train.py`, then upload the whole folder to a GitHub repo (include `model.joblib`, `outputs/`, `.streamlit/`).
2. Go to https://share.streamlit.io, sign in with GitHub, choose the repo and set the main file to `app.py`.
3. You get a URL that opens on any device. On a phone: browser menu, "Add to Home Screen" gives an app-like icon.

**Option 2: same Wi-Fi only** (good for the live demo at the university):
```bash
streamlit run app.py --server.address 0.0.0.0
```
Then open `http://<your-laptop-IP>:8501` from any phone or tablet on the same network.

## Notes
- The app reads the needed columns from the trained model itself, so it also works with a smaller model (for example one trained on only 6 features).
- The extra fields (age, family, social) are optional; sensible defaults are used when they are left alone.

## Pipeline
1. **EDA**: target distribution, correlation matrix, key-feature boxplots
2. **Preprocessing**: StandardScaler (numeric) + OneHotEncoder (categorical) inside a scikit-learn `Pipeline`
3. **Models**: Logistic Regression, Random Forest, Gradient Boosting (5-fold stratified CV, F1)
4. **Evaluation**: accuracy, F1, ROC-AUC, confusion matrix, ROC curves
5. **Interpretation**: permutation importance
6. **Deployment**: Streamlit app

## Structure
```
train.py            training + evaluation
app.py              Streamlit app (responsive, 3 languages)
i18n.py             translations (ar / en / fr)
logic.py            prediction logic and tips (no UI, testable)
.streamlit/         theme and server settings
data/               put student-mat.csv here
outputs/            plots + metrics.json (generated)
model.joblib        trained model (generated)
```

## Ideas to extend (for extra marks)
- Hyperparameter tuning with `GridSearchCV`
- Handle class imbalance (`class_weight="balanced"`, SMOTE)
- SHAP explanations per student
- Regression version predicting the exact grade
- Add the Portuguese-course dataset and compare
-