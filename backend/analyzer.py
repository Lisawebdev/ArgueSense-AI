"""
analyzer.py — the machine learning core of Argument Analyzer.

This file never talks to the network. It is pure Python + scikit-learn and
can be imported, run, and tested completely on its own.

It holds:
  - TRAIN_SENTENCES: 50 hand-labelled example sentences (25 strong, 25 weak),
    each with a hand-assigned strength score between 0.10 and 0.92.
  - ArgumentModel: trains a TF-IDF vectorizer, an SVM classifier, a KNN
    classifier (for comparison), an SVR regressor, and (per-request) a
    K-Means clusterer, then exposes methods to analyze new text.
"""

import re
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR

# ---------------------------------------------------------------------------
# 1. Training data: 50 tuples of (sentence_text, label, strength_score)
#    label: 1 = strong argument, 0 = weak argument
#    strength_score: hand-assigned, 0.10 - 0.92
# ---------------------------------------------------------------------------

TRAIN_SENTENCES = [
    # ---- STRONG (label = 1) ----
    ("Peer-reviewed studies from three independent universities confirm that the drug reduces relapse rates by over 40 percent.", 1, 0.92),
    ("The city's own budget report shows a 15 percent drop in emergency response times after the new fire stations opened.", 1, 0.90),
    ("Historical records from the national archive directly contradict the defendant's claimed timeline for that evening.", 1, 0.88),
    ("A randomized controlled trial involving 12,000 participants found no statistically significant difference between the two treatments.", 1, 0.91),
    ("Government census data shows household income in the region has risen steadily for six consecutive years.", 1, 0.86),
    ("The bridge's structural engineers published stress-test results showing it can safely bear twice its rated load.", 1, 0.89),
    ("Independent audits of the company's finances found no evidence of the fraud alleged in the lawsuit.", 1, 0.87),
    ("Satellite temperature measurements collected over four decades show a consistent warming trend across every continent.", 1, 0.90),
    ("The university's longitudinal study tracked the same 500 students for a decade and found tutoring raised graduation rates by 22 percent.", 1, 0.88),
    ("Court transcripts confirm the witness gave two contradictory accounts of the same event under oath.", 1, 0.85),
    ("A meta-analysis combining results from 40 separate clinical trials found the vaccine reduced hospitalization by 85 percent.", 1, 0.92),
    ("The manufacturer's own recall filing lists the exact defect that caused the reported engine fires.", 1, 0.84),
    ("Economic data from the central bank shows inflation cooling for four straight quarters after the policy change.", 1, 0.83),
    ("Forensic analysis of the soil samples places the vehicle at the scene within a two-hour window.", 1, 0.86),
    ("The school district's test scores rose every year after the new reading curriculum was introduced.", 1, 0.80),
    ("Multiple independent fact-checking organizations traced the viral claim back to a single fabricated source.", 1, 0.85),
    ("The hospital's infection rate dropped by half after mandatory handwashing protocols were enforced.", 1, 0.82),
    ("Publicly available flight records show the executive's plane never landed anywhere near the alleged meeting.", 1, 0.87),
    ("A controlled field experiment across 30 farms found the new fertilizer increased crop yield by 18 percent.", 1, 0.84),
    ("The insurance company's own actuarial tables demonstrate that the risk was known well before the policy was sold.", 1, 0.83),
    ("Body camera footage directly shows the officer's account of the incident was inaccurate.", 1, 0.86),
    ("National employment statistics confirm that manufacturing jobs in the region grew for the first time in a decade.", 1, 0.81),
    ("The lab's replicated experiment produced the same result in nine out of ten trials, matching the original study.", 1, 0.85),
    ("Utility company billing records show the average customer's rate did not change despite the new surcharge claim.", 1, 0.79),
    ("A systematic review of workplace safety data found that mandatory breaks cut injury rates by nearly a third.", 1, 0.83),

    # ---- WEAK (label = 0) ----
    ("Everyone knows that this kind of thing just doesn't work out in the end.", 0, 0.18),
    ("My cousin tried it once and said it was basically useless.", 0, 0.20),
    ("It just feels like the right thing to do, so it probably is.", 0, 0.15),
    ("I saw a post online that said the same thing, so it must be true.", 0, 0.22),
    ("Honestly, who even trusts these so-called experts anyway.", 0, 0.12),
    ("Back in the day things were just better, everybody says so.", 0, 0.17),
    ("If it were really a problem, someone would have already fixed it by now.", 0, 0.24),
    ("A friend of a friend heard that the company is basically shutting down soon.", 0, 0.19),
    ("This is obviously the only reasonable way to look at the situation.", 0, 0.21),
    ("People have always done it this way, so changing it now seems pointless.", 0, 0.23),
    ("I just have a gut feeling that the numbers are being exaggerated.", 0, 0.16),
    ("Some guy on TV said the opposite, so who really knows what's true.", 0, 0.18),
    ("It sounds fake to me, but I can't really explain why.", 0, 0.14),
    ("Nobody I know has ever had a problem with it, so it must be fine for everyone.", 0, 0.25),
    ("That's just common sense, you don't need a study to tell you that.", 0, 0.20),
    ("I read somewhere that this happens all the time, though I forget where.", 0, 0.22),
    ("It's probably a conspiracy anyway, these things usually are.", 0, 0.13),
    ("If the product were actually bad, it wouldn't still be sold in stores.", 0, 0.26),
    ("I feel like the whole thing was probably exaggerated by the media.", 0, 0.19),
    ("Everybody at my job agrees, so it has to be a widespread issue.", 0, 0.21),
    ("It's just obvious once you think about it for a second.", 0, 0.15),
    ("I don't trust the report because it just doesn't match my experience.", 0, 0.24),
    ("This trend will probably keep going forever, it always has before.", 0, 0.18),
    ("My neighbor swears by it, and that's good enough proof for me.", 0, 0.20),
    ("It can't be that serious, or we would have heard way more about it.", 0, 0.23),
]

# Minimum length (in characters) for a sentence fragment to be kept by the
# splitter. Anything shorter is discarded as noise (stray punctuation,
# abbreviations, etc). app.py enforces the same minimum on raw input text.
MIN_SENTENCE_LENGTH = 15


class ArgumentModel:
    """Holds all four trained models and the pipeline that uses them."""

    def __init__(self):
        texts = [s[0] for s in TRAIN_SENTENCES]
        labels = np.array([s[1] for s in TRAIN_SENTENCES])
        scores = np.array([s[2] for s in TRAIN_SENTENCES])

        # --- TF-IDF vectorizer, fit once on the training sentences ---
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 1),
            max_features=150,
        )
        X = self.vectorizer.fit_transform(texts)

        # --- 5-fold cross-validation for SVM and KNN (evaluation only) ---
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        svm_cv = SVC(kernel="linear")
        knn_cv = KNeighborsClassifier(n_neighbors=3)

        self.svm_fold_scores = cross_val_score(svm_cv, X, labels, cv=skf).tolist()
        self.knn_fold_scores = cross_val_score(knn_cv, X, labels, cv=skf).tolist()

        self.svm_cv_accuracy = float(np.mean(self.svm_fold_scores))
        self.knn_cv_accuracy = float(np.mean(self.knn_fold_scores))

        # --- Fit final models on all 50 sentences ---
        self.svm = SVC(kernel="linear", probability=False)
        self.svm.fit(X, labels)

        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(X, labels)

        self.svr = SVR(kernel="linear")
        self.svr.fit(X, scores)

        self.train_size = len(TRAIN_SENTENCES)
        self.strong_count = int(labels.sum())
        self.weak_count = self.train_size - self.strong_count

        print(f"Done. SVM CV accuracy: {self.svm_cv_accuracy:.2f}")

    # -----------------------------------------------------------------
    def metrics(self):
        """Training set size, class balance, and CV accuracy for both
        classifiers (mean and per-fold), so the dashboard can show real,
        live numbers instead of anything hard-coded."""
        return {
            "train_size": self.train_size,
            "strong_count": self.strong_count,
            "weak_count": self.weak_count,
            "svm": {
                "cv_accuracy": round(self.svm_cv_accuracy, 4),
                "fold_scores": [round(s, 4) for s in self.svm_fold_scores],
            },
            "knn": {
                "cv_accuracy": round(self.knn_cv_accuracy, 4),
                "fold_scores": [round(s, 4) for s in self.knn_fold_scores],
            },
            "served_by": "svm",
        }

    # -----------------------------------------------------------------
    @staticmethod
    def split_into_sentences(text):
        """Small regex-based sentence splitter. No external NLP download
        required. Splits on ., !, or ? followed by whitespace, then drops
        any fragment shorter than MIN_SENTENCE_LENGTH characters."""
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in raw if len(s.strip()) >= MIN_SENTENCE_LENGTH]
        return sentences

    # -----------------------------------------------------------------
    def cluster_sentences(self, sentences, n_clusters):
        """Runs K-Means on a *new* batch of sentences (never the training
        set) and labels each cluster with its most distinctive word, taken
        from the cluster center in TF-IDF space."""
        n_clusters = max(1, min(n_clusters, len(sentences)))

        X = self.vectorizer.transform(sentences)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_ids = km.fit_predict(X)

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        cluster_labels = {}
        for cid in range(n_clusters):
            center = km.cluster_centers_[cid]
            if center.sum() == 0 or len(feature_names) == 0:
                cluster_labels[cid] = f"theme {cid + 1}"
            else:
                top_idx = int(np.argmax(center))
                cluster_labels[cid] = feature_names[top_idx]

        return cluster_ids, cluster_labels

    # -----------------------------------------------------------------
    def analyze_essay(self, text, n_clusters=3):
        """Full pipeline for new text: split -> vectorize -> classify
        (SVM) -> score (SVR) -> cluster (K-Means, refit per request) ->
        structured results."""
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        X = self.vectorizer.transform(sentences)

        labels = self.svm.predict(X)
        scores = self.svr.predict(X)
        scores = np.clip(scores, 0.0, 1.0)

        cluster_ids, cluster_labels = self.cluster_sentences(sentences, n_clusters)

        results = []
        for i, sentence in enumerate(sentences):
            results.append({
                "sentence": sentence,
                "label": "STRONG" if labels[i] == 1 else "WEAK",
                "strength_score": round(float(scores[i]), 3),
                "theme": cluster_labels[int(cluster_ids[i])],
            })
        return results


if __name__ == "__main__":
    # Quick self-test: runnable on its own, no Flask required.
    model = ArgumentModel()
    print(model.metrics())
    sample = (
        "A randomized trial of 2,000 patients found the therapy cut recovery "
        "time in half. Honestly, everyone I know just says it works better. "
        "The hospital's own records confirm the same drop in readmission rates."
    )
    for r in model.analyze_essay(sample, n_clusters=2):
        print(r)
