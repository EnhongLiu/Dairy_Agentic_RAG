import pandas as pd
import re
from langchain.schema import Document

# Read in data
jds_metadata = pd.read_csv('./jds_metadata.csv')
animal_metadata = pd.read_csv('./animal_metadata.csv')


# Cleaning to drop rows with missing abstracts
jds_cleaned = jds_metadata.copy().dropna(subset=['Abstract'])
animal_cleaned = animal_metadata.copy().dropna(subset=['Abstract'])

def extract_abstract(text):
    match = re.search(r"Author information:.*?\n\n(.*?)\n\n(?:Copyright|DOI|PMID)", text, re.DOTALL)
    if match:
        abstract = match.group(1).strip()
        abstract = re.sub(r'\s+', ' ', abstract)
        return abstract
    return None

jds_cleaned['Pure Abstract'] = jds_cleaned['Abstract'].apply(extract_abstract)
animal_cleaned['Pure Abstract'] = animal_cleaned['Abstract'].apply(extract_abstract)


# Combine JDS and Animal
journal_df = pd.concat([jds_cleaned,animal_cleaned])[['Title','Pure Abstract','Authors','Publication Date','DOI','Link','Affiliations','PubMed ID','Publication Year','Country']]

journal_df_cleaned = journal_df.dropna(subset=['Pure Abstract'], how='any')

# Create Document objects
documents = [
    Document(
        page_content=row["Pure Abstract"],
        metadata={key: value for key, value in row.items() if key != "Pure Abstract"}
    )
    for _, row in journal_df_cleaned.iterrows()
]
