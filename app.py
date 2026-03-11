import streamlit as st
import math
import pandas as pd
from indicnlp.tokenize import indic_tokenize
import string

# --- CORE LOGIC (TF-IDF) ---
def calculate_tfidf(documents, stop_words):
    # 1. Improved Tokenization: Remove punctuation and empty strings
    tokenized_docs = []
    for doc in documents:
        # Remove punctuation like '-' or '.'
        clean_text = doc.translate(str.maketrans('', '', string.punctuation))
        tokens = [t for t in indic_tokenize.trivial_tokenize(clean_text) if t.strip()]
        tokenized_docs.append(tokens)
        
    all_words = set(word for doc in tokenized_docs for word in doc)
    
    # 2. TF Calculation
    tf_scores = []
    for doc in tokenized_docs:
        doc_tf = {word: (doc.count(word) / len(doc) if len(doc) > 0 else 0) for word in all_words}
        tf_scores.append(doc_tf)
    
    # 3. IDF Calculation
    idf_scores = {}
    total_docs = len(documents)
    for word in all_words:
        docs_with_word = sum(1 for doc in tokenized_docs if word in doc)
        # Adding 1 to denominator to prevent division by zero
        idf_scores[word] = math.log(total_docs / (1 + docs_with_word))
        
    # 4. Final Scoring with Filter
    final_results = []
    for doc_tf in tf_scores:
        # FILTER: Ignore stop words AND words that are too short (like single letters)
        doc_tfidf = {word: doc_tf[word] * idf_scores[word] 
                     for word in all_words 
                     if word not in stop_words and len(word) > 1}
        
        sorted_keywords = sorted(doc_tfidf.items(), key=lambda x: x[1], reverse=True)
        final_results.append(sorted_keywords)
        
    return final_results

# --- STREAMLIT UI ---
st.set_page_config(page_title="Indic NLP Extractor", page_icon="✍️")

st.title("✍️ Multilingual Keyword Extractor")
st.markdown("Extract important keywords from **Marathi** and **Kannada** text using TF-IDF.")

# Sidebar for Stop-words (Crucial for recruiters to see you handle noise)
st.sidebar.header("Settings")
user_stop_words = st.sidebar.text_area("Stop-words (comma separated)", "आणि, मी, आहे, हे, ನನಗೆ, ಮತ್ತು, ಹಾಗೂ").split(", ")

# Input Area
st.subheader("Enter your Documents")
doc1 = st.text_area("Document 1 (e.g., Marathi)", "मला प्रोग्रामिंग आवडते आणि मी एआय शिकत आहे")
doc2 = st.text_area("Document 2 (e.g., Kannada)", "ನನಗೆ ಪ್ರೋಗ್ರಾಮಿಂಗ್ ಇಷ್ಟ ಮತ್ತು ನಾನು ಎಐ ಕಲಿಯುತ್ತಿದ್ದೇನೆ")
doc3 = st.text_area("Document 3", "एआय हे भविष्यातील तंत्रज्ञान आहे")

if st.button("Extract Keywords"):
    docs = [doc1, doc2, doc3]
    results = calculate_tfidf(docs, user_stop_words)
    
    cols = st.columns(3)
    for i, keywords in enumerate(results):
        with cols[i]:
            st.success(f"Doc {i+1} Keywords")
            # Filter for scores > 0
            valid_k = [k for k in keywords if k[1] > 0][:5]
            if valid_k:
                for word, score in valid_k:
                    st.write(f"**{word}** (Score: {score:.2f})")
            else:
                st.write("No unique keywords found.")

st.info("Logic: Custom TF-IDF implementation using Indic-NLP Tokenization.")