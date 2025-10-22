import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Cleans a single text string:
    - Removes non-alphabetic characters
    - Converts to lowercase
    - Removes stopwords
    - Lemmatizes words
    """
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def load_and_preprocess_data(files_labels_cols):
    """
    Loads multiple CSV files, cleans their text, and labels them.

    Parameters:
        files_labels_cols (list of tuples): [(file_path, label, text_column), ...]
            label: 0 for fake, 1 for real
            text_column: name of the column containing text in this CSV (can be None)
    Returns:
        pd.DataFrame: combined and cleaned data
    """
    all_data = []

    for file_path, label, text_col in files_labels_cols:
        df = pd.read_csv(file_path)

        # Auto-detect text column if None
        if text_col is None:
            # Pick the first column with object (string) type
            text_col_candidates = df.select_dtypes(include=['object']).columns
            if len(text_col_candidates) == 0:
                raise ValueError(f"No text column found in {file_path}")
            text_col = text_col_candidates[0]
            print(f"ℹ️ Auto-detected text column '{text_col}' in {file_path}")

        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in {file_path}")

        df['label'] = label
        df['clean_text'] = df[text_col].apply(clean_text)
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    return combined
