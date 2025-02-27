import PyPDF2
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download NLTK data (only needed once)
nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def preprocess_text(text):
    """Convert text to lowercase, tokenize, and remove stopwords and non-alphabetic tokens."""
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words_filtered = [word for word in words if word.isalpha() and word not in stop_words]
    return words_filtered

def get_top_keywords(words, num_keywords=10):
    """Return the most common keywords from the list of words."""
    frequency = Counter(words)
    return frequency.most_common(num_keywords)

def get_contexts(text, keyword):
    """Find sentences in the text that contain the given keyword."""
    sentences = sent_tokenize(text)
    keyword_contexts = [sentence.strip() for sentence in sentences if keyword in sentence.lower()]
    return keyword_contexts

def main(pdf_path, num_keywords=10):
    # Extract and preprocess text
    text = extract_text_from_pdf(pdf_path)
    words_filtered = preprocess_text(text)

    # Get the top keywords
    top_keywords = get_top_keywords(words_filtered, num_keywords)
    print("Top Keywords and Their Occurrence Counts:")
    for keyword, count in top_keywords:
        print(f"{keyword}: {count}")

    print("\nKeyword Contexts:")
    # Print context for each keyword
    for keyword, count in top_keywords:
        print(f"\nKeyword: '{keyword}' (Count: {count})")
        contexts = get_contexts(text, keyword)
        if contexts:
            for context in contexts:
                print(f" - {context}")
        else:
            print(" - No context found.")

if __name__ == "__main__":
    # Replace 'document.pdf' with the path to your PDF file.
    pdf_file = "document.pdf"
    main(pdf_file)
