# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset dari file lokal
df = pd.read_csv("preprocessed_dataset.csv")  # Ganti dengan nama file CSV kamu

# Pisahkan fitur dan label
X = df['Teks']
y = df['label']

# Vektorisasi teks
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Grid Search untuk parameter SVM
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1]
}
svm = SVC(kernel='rbf', random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluasi
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Simpan model dan vectorizer ke satu file
bundle = {
    "model": best_model,
    "vectorizer": vectorizer
}
joblib.dump(bundle, "spam_model_bundle.pkl")
print("Model dan vectorizer disimpan ke spam_model_bundle.pkl")
