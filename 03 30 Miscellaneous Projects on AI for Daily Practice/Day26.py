# Install spaCy and download model if needed
# !pip install spacy
# !python -m spacy download en_core_web_sm

# Import necessary libraries
import spacy
from spacy import displacy
import pandas as pd

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Sample text for NER
text = """
Amazon announced its quarterly earnings on July 30, 2023. 
CEO Andy Jassy said the company is investing $4 billion in AI technology. 
Google, based in Mountain View, California, also shared its financial report. 
The 2024 Summer Olympics will be held in Paris, France.
"""

# Process the text with spaCy
doc = nlp(text)

# Function to extract entities
def extract_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append({
            'Entity': ent.text,
            'Label': ent.label_,
            'Explanation': spacy.explain(ent.label_)
        })
    return pd.DataFrame(entities)

# Extract entities into a DataFrame
entities_df = extract_entities(doc)

# Display extracted entities to the user
print("Extracted Named Entities:")
print(entities_df)

# Visualize Named Entities using DisplaCy
displacy.render(doc, style="ent", jupyter=True)

# Save entities to a CSV file
entities_df.to_csv("extracted_entities.csv", index=False)
print("\nEntities saved to 'extracted_entities.csv'")
