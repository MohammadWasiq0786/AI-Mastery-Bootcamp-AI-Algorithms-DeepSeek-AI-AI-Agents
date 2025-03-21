# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download stopwords (only needed once)
nltk.download("stopwords")
nltk.download("punkt")

# Example text for summarization
text = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. 
The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.

The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal. 
A subset of artificial intelligence is machine learning, which refers to the concept that computer programs can automatically learn from and adapt to new data without being assisted by humans. 
Deep learning techniques enable this automatic learning through the absorption of huge amounts of unstructured data such as text, images, or video.
"""

# Function to generate a frequency-based summary
def summarize_text(text, num_sentences=2):
    # Tokenize text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Filter out stopwords and non-alphabetic words
    stop_words = set(stopwords.words("english"))
    word_frequencies = {}
    
    for word in words:
        if word.isalpha() and word not in stop_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    # Score each sentence based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]

    # Sort sentences by score and select the top `num_sentences`
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = " ".join(summary_sentences)
    return summary

# Generate and print the summary
summary = summarize_text(text, num_sentences=2)
print("Original Text:\n", text)
print("\nSummary:\n", summary)
