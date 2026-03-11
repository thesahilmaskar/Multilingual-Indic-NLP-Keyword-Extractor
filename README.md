# ✍️ Multilingual Indic-NLP Keyword Extractor

A high-performance NLP pipeline built to extract meaningful keywords from **Marathi**, **Kannada**, and **English** text using a custom-implemented **TF-IDF (Term Frequency-Inverse Document Frequency)** algorithm.

---

## 🧠 The Project Goal
As a final-year AIML student, I developed this project to bridge the gap between theoretical Information Retrieval and practical, multilingual applications. Most standard libraries are optimized for English; this tool specifically handles the morphological complexities of Indic scripts.

### Key Features:
- **Custom TF-IDF Engine:** Built from scratch to demonstrate mathematical understanding of NLP vectorization.
- **Indic-NLP Integration:** Uses specific tokenization for Marathi and Kannada to handle complex script markers.
- **Noise Reduction:** Advanced preprocessing to filter out punctuation and language-specific stop-words.
- **Interactive Dashboard:** Built with Streamlit for real-time text analysis.

---

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **NLP:** Indic-NLP-Library, NLTK
- **Math/Data:** NumPy, Pandas, Math API
- **UI/UX:** Streamlit
- **Deployment:** GitHub & Streamlit Cloud

---

## 📊 How It Works (The Logic)
1. **Tokenization:** Text is broken into tokens using script-aware logic.
2. **TF Calculation:** Measures word frequency in a specific document.
3. **IDF Calculation:** Weights the word's uniqueness across the entire corpus.
4. **Filtering:** Removes common functional words (stop-words) to reveal the "thematic" core of the text.



---

## 📈 Future Scope
- **Stemming/Lemmatization:** Integrating a root-word extractor for Marathi/Kannada to group different word forms (e.g., "विकासाचा" and "विकास").
- **API Support:** Converting the logic into a FastAPI endpoint for integration into larger ERP systems (like SAP ABAP environments).
- **Word Embeddings:** Moving from statistical TF-IDF to semantic models like **IndicBERT**.

---

## 👤 Author
**Sahil Maskar** Final Year B.E. (AIML) | SIES GST  
*Interested in Web Development & Machine Learning*

