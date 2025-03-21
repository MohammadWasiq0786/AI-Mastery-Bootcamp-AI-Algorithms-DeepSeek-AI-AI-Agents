# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample resumes and job description data
data = {
    'resume_id': [1, 2, 3],
    'resume_text': [
        "Experienced data scientist with skills in Python, machine learning, and data analysis.",
        "Software developer with expertise in Java, cloud computing, and project management.",
        "Data analyst with proficiency in SQL, Python, and data visualization."
    ]
}

job_description = "Looking for a data scientist skilled in Python, machine learning, SQL, and data analysis."

# Convert to DataFrame
df = pd.DataFrame(data)
print("Resumes:\n", df)

# Combine job description with resumes for TF-IDF vectorization
documents = df['resume_text'].tolist()
documents.append(job_description)

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate similarity scores between job description and each resume
similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

# Display similarity scores for each resume
df['similarity_score'] = similarity_scores
print("\nResume Similarity Scores:\n", df[['resume_id', 'similarity_score']])

# Identify resumes that match the job requirements (threshold can be adjusted)
threshold = 0.2
matching_resumes = df[df['similarity_score'] >= threshold]
print("\nResumes matching the job requirements:\n", matching_resumes[['resume_id', 'similarity_score']])
