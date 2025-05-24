# AI-Powered Resume Ranker

## Objective
Automatically rank resumes based on their relevance to a job description using NLP techniques like SpaCy and TF-IDF.

---

## Features
- Extracts text from PDF resumes
- Preprocesses text using **SpaCy**
- Converts text into **TF-IDF vectors**
- Calculates similarity scores to the job description
- Ranks candidates based on relevance
- Web UI for uploading resumes and viewing results
- HR-friendly reports available for download

---

## Tools used
- Python
- Flask (Web App)
- SpaCy (NLP)
- Scikit-learn (TF-IDF)
- PyPDF2 / pdfminer.six (PDF parsing)
- HTML/CSS (Frontend)

---

## Project Structure
resume-ranker/
├── app.py # Flask app
├── utils.py # Helper functions (NLP, scoring, etc.)
├── templates/
│ └── index.html # Web UI
├── static/
│ └── style.css # Optional styling
├── resumes/ # Uploaded PDF resumes
├── ranked_reports/ # Output reports (CSV)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## How It Works

1. Upload one or more PDF resumes via the web UI.
2. Input or paste the job description.
3. The app:
   - Extracts and cleans text using SpaCy.
   - Converts text into TF-IDF vectors.
   - Compares each resume against the job description.
   - Scores and ranks each candidate.
4. View the ranked list and download an HR report (CSV).
---
## Sample Output
| Rank | Candidate    | Score |
| ---- | ------------ | ----- |
| 1    | John Doe     | 0.92  |
| 2    | Jane Smith   | 0.87  |
| 3    | Alex Johnson | 0.76  |


