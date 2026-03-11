import math
import pandas as pd
from indicnlp.tokenize import indic_tokenize

def extract_keywords(documents, top_n=3):
    # 1. Tokenization
    tokenized_docs = [indic_tokenize.trivial_tokenize(doc) for doc in documents]
    all_words = set(word for doc in tokenized_docs for word in doc)
    
    # 2. Calculate Term Frequency (TF)
    tf_scores = []
    for doc in tokenized_docs:
        doc_tf = {}
        for word in all_words:
            # TF = frequency of word / total words in that doc
            doc_tf[word] = doc.count(word) / len(doc) if len(doc) > 0 else 0
        tf_scores.append(doc_tf)
    
    # 3. Calculate Inverse Document Frequency (IDF)
    idf_scores = {}
    total_docs = len(documents)
    for word in all_words:
        docs_with_word = sum(1 for doc in tokenized_docs if word in doc)
        # IDF = log(Total Docs / Docs containing the word)
        # We add 1 to the denominator to avoid division by zero errors
        idf_scores[word] = math.log(total_docs / (1 + docs_with_word))
        
    # 4. Calculate Final TF-IDF
    tfidf_results = [] # <--- Corrected name here
    for doc_tf in tf_scores:
        doc_tfidf = {word: doc_tf[word] * idf_scores[word] for word in all_words}
        # Sort words by their TF-IDF score in descending order
        sorted_keywords = sorted(doc_tfidf.items(), key=lambda x: x[1], reverse=True)
        tfidf_results.append(sorted_keywords[:top_n])
        
    return tfidf_results # <--- Changed from tf_results to tfidf_results

# --- Execution ---
sample_docs = [
    "मला प्रोग्रामिंग आवडते आणि मी एआय शिकत आहे", 
    "ನನಗೆ ಪ್ರೋಗ್ರಾಮಿಂಗ್ ಇಷ್ಟ ಮತ್ತು ನಾನು ಎಐ ಕಲಿಯುತ್ತಿದ್ದೇನೆ", 
    "एआय हे भविष्यातील तंत्रज्ञान आहे" 
]

results = extract_keywords(sample_docs)

print("\n--- Multilingual Keyword Results ---")
for i, keywords in enumerate(results):
    # Filter out keywords with 0 score (meaning they appear in every doc)
    valid_keywords = [k[0] for k in keywords if k[1] > 0]
    print(f"Document {i+1}: {', '.join(valid_keywords)}")