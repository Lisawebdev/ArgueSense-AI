# Argument Analyzer

A full-stack web app that reads an essay, article, or debate transcript, splits it
into sentences, and for each one predicts:

1. **STRONG or WEAK** — a classification decision (SVM)
2. **A 0–1 strength score** — a regression, so two "strong" sentences can still be
   ranked against each other (SVR)
3. **A theme/topic**, grouped against the other sentences in the same text — an
   unsupervised clustering decision (K-Means)

Nothing is faked or pre-recorded — every number the app shows is produced by the
models at the moment it's requested.

The app has three views:
- **Landing** — title and a "Get Started" button.
- **Analyze** — paste text in, get per-sentence results back.
- **Settings** — live accuracy metrics, the 5-fold cross-validation chart, and a
  plain-language explanation of how each algorithm is used.

---

## Project structure

```
argument_analyzer_app/
├── backend/
│   ├── analyzer.py         # ML core: training data + SVM, KNN, SVR, K-Means
│   ├── app.py                # Flask server: trains once, exposes the API
│   ├── requirements.txt
│   └── Procfile              # start command for hosting platforms
├── frontend/
│   └── index.html             # landing + analyze + settings, one file, no build step
├── .gitignore
└── README.md
```

---

## Run it locally

1. Make sure Python 3.9+ is installed.
2. ```bash
   cd argument_analyzer_app/backend
   python -m venv venv
   venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt
   python app.py
   ```
3. You should see:
   ```
   Training models (SVM, KNN, SVR, K-Means)...
   Done. SVM CV accuracy: 0.86
    * Running on http://127.0.0.1:5000
   ```
4. Open **http://127.0.0.1:5000** in a browser. Do not open `frontend/index.html`
   directly from disk — it has no data of its own and needs the Flask server behind it.

---

## API

- `GET /api/metrics` — training set size, class balance, and 5-fold cross-validation
  accuracy (mean + per-fold) for both SVM and KNN.
- `POST /api/analyze` — body `{"text": "...", "n_clusters": 3}`. Text must be at
  least 15 characters (shorter input can't produce a meaningful sentence). Returns
  a list of `{sentence, label, strength_score, theme}` objects.

---

## Notes

- Training happens once, at server startup, on 50 fixed hand-labelled sentences.
  User-submitted text is never used to retrain the model — only to run through it.
- If you edit `analyzer.py`, restart the server (`Ctrl+C`, then `python app.py`
  again) for the change to take effect.
- `n_clusters` is automatically capped at the number of sentences found in the
  submitted text, since K-Means can't form more clusters than there are data points.

---

## Push to GitHub

From inside the `argument_analyzer_app` folder (the `.gitignore` here already
excludes `venv/` and `__pycache__/`):

```bash
git init
git add .
git commit -m "Initial commit: Argument Analyzer"
git branch -M main
git remote add origin https://github.com/yourusername/argument-analyzer.git
git push -u origin main
```

Create the empty repo on GitHub first (https://github.com/new) and swap in its URL.
For later changes:

```bash
git add .
git commit -m "describe what changed"
git push
```

---

## Deploy it live (Render)

The `Procfile` and environment-configurable port in `app.py` are already set up
for this.

1. Push the repo to GitHub (above).
2. Go to https://render.com, sign in with GitHub, click **New +** → **Web Service**,
   and select this repo.
3. Configure it:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free
4. Click **Create Web Service**. Render installs dependencies, trains the models
   on startup, and gives you a public URL like `https://argument-analyzer.onrender.com`.

The free tier spins down after inactivity and takes ~30–50 seconds to wake back up
on the next visit — that's normal, not a bug.
