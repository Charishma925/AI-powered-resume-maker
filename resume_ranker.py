import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# File to save ranked resumes
CSV_FILENAME = "ranked_resumes.csv"

# --- Step 1: Extract text from PDF resumes ---
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text

# --- Step 2: Preprocess text using SpaCy ---
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# --- Step 3: Extract emails and names using SpaCy NER and regex ---
def extract_entities(text):
    doc = nlp(text)
    emails = re.findall(r'\S+@\S+', text)
    names = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    return emails, names

# --- Main ranking function ---
def rank_resumes(resume_paths, job_description):
    # Preprocess job description
    job_desc_clean = preprocess_text(job_description)
    
    # Initialize TF-IDF vectorizer on job description only
    tfidf_vectorizer = TfidfVectorizer()
    job_desc_vector = tfidf_vectorizer.fit_transform([job_desc_clean])
    
    ranked_resumes = []
    
    for resume_path in resume_paths:
        # Extract and preprocess resume text
        resume_text = extract_text_from_pdf(resume_path)
        resume_text_clean = preprocess_text(resume_text)
        
        # Extract entities
        emails, names = extract_entities(resume_text)
        
        # Vectorize resume
        resume_vector = tfidf_vectorizer.transform([resume_text_clean])
        
        # Calculate similarity score
        similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0]
        
        ranked_resumes.append({
            "name": names[0] if names else "N/A",
            "email": emails[0] if emails else "N/A",
            "similarity": similarity,
            "file": resume_path
        })
    
    # Sort by similarity descending
    ranked_resumes.sort(key=lambda x: x["similarity"], reverse=True)
    
    return ranked_resumes

# --- Step 7: Save results to CSV ---
def save_ranked_to_csv(ranked_resumes, filename=CSV_FILENAME):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Rank", "Name", "Email", "Similarity", "Resume File"])
        for rank, resume in enumerate(ranked_resumes, start=1):
            writer.writerow([rank, resume["name"], resume["email"], f"{resume['similarity']:.4f}", resume["file"]])

# --- Example usage ---
if __name__ == "__main__":
    # Example job description
    job_description = """NLP Specialist: Develop and implement NLP algorithms. Proficiency in Python, NLP libraries, and ML frameworks required."""
    
    # List your PDF resume paths here
    resume_files = ["resume1.pdf", "resume2.pdf", "resume3.pdf"]
    
    # Rank resumes
    ranked = rank_resumes(resume_files, job_description)
    
    # Print ranked resumes
    for idx, res in enumerate(ranked, start=1):
        print(f"Rank {idx}: {res['name']}, {res['email']}, Similarity: {res['similarity']:.4f}")
    
    # Save results to CSV
    save_ranked_to_csv(ranked)
    print(f"\nRanking saved to {CSV_FILENAME}")
