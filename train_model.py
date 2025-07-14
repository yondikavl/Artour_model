import pandas as pd
import joblib
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df['Teks'], df['label']


def vectorize_text(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    return vectorizer, vectorizer.fit_transform(texts)


def train_svm(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1]
    }
    svm = SVC(kernel='rbf', random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True)
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Akurasi:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Bar chart untuk metrik
    metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}
    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.ylim(0, 1.1)
    plt.title('Evaluation Metrics')
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.show()


def visualize_per_fold_scores(grid_search):
    results = pd.DataFrame(grid_search.cv_results_)
    best_params = grid_search.best_params_

    mask = (results['param_C'] == best_params['C']) & (results['param_gamma'] == best_params['gamma'])
    best_row = results[mask].iloc[0]

    fold_scores = [best_row[f'split{i}_test_score'] for i in range(grid_search.cv)]
    folds = list(range(1, grid_search.cv + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(folds, fold_scores, marker='o', linestyle='-', color='purple')
    plt.xticks(folds)
    plt.ylim(0, 1.05)
    plt.title(f"Accuracy per Fold (Best Params: C={best_params['C']}, gamma={best_params['gamma']})")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    for i, score in enumerate(fold_scores):
        plt.text(folds[i], score + 0.01, f"{score:.3f}", ha='center')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_label_distribution_per_fold(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_data = []
    for fold, (_, test_index) in enumerate(skf.split(X, y), start=1):
        y_fold = y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]
        count_0 = (y_fold == 0).sum()
        count_1 = (y_fold == 1).sum()
        fold_data.append({'Fold': fold, 'Label': 'Non-Spam', 'Count': count_0})
        fold_data.append({'Fold': fold, 'Label': 'Spam', 'Count': count_1})

    df_fold = pd.DataFrame(fold_data)

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df_fold, x='Fold', y='Count', hue='Label', palette='Set2')

    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', padding=3)

    plt.title("Distribusi Label Spam/Non-Spam per Fold")
    plt.ylabel("Jumlah Sampel")
    plt.tight_layout()
    plt.show()


def visualize_top_words_by_class(texts, labels, vectorizer, top_n=20):
    spam_texts = texts[np.array(labels) == 1]
    nonspam_texts = texts[np.array(labels) == 0]

    features = vectorizer.get_feature_names_out()

    def plot_top_words(text_matrix, class_label):
        word_freq = np.asarray(text_matrix.sum(axis=0)).flatten()
        word_counts = dict(zip(features, word_freq))
        top_words = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])

        plt.figure(figsize=(10, 5))
        plt.bar(top_words.keys(), top_words.values(), color='salmon' if class_label == 'Spam' else 'skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Top {top_n} Words in {class_label} Messages (TF-IDF)")
        plt.ylabel("Frequency (TF-IDF Score)")
        plt.tight_layout()
        plt.show()

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Word Cloud for {class_label} Messages")
        plt.show()

    plot_top_words(nonspam_texts, "Non-Spam")
    plot_top_words(spam_texts, "Spam")


def save_model(model, vectorizer, filename):
    bundle = {
        "model": model,
        "vectorizer": vectorizer
    }
    joblib.dump(bundle, filename)


if __name__ == "__main__":
    # Load dan preprocessing
    X, y = load_dataset("preprocessed_dataset.csv")
    vectorizer, X_vect = vectorize_text(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

    # Training + cross-validation
    grid_search = train_svm(X_train, y_train)
    model = grid_search.best_estimator_

    # Evaluasi & visualisasi
    evaluate_model(model, X_test, y_test)
    # visualize_per_fold_scores(grid_search)
    # visualize_label_distribution_per_fold(X_vect, y)
    # visualize_top_words_by_class(X_vect, y, vectorizer)

    # Simpan model
    save_model(model, vectorizer, "spam_model_filter.pkl")
