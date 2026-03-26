# 💊 DrugSentiment_POS_NER

![App Screenshot](screenshot.png)

## 📌 Project Pipeline

### 1. Setup
Import libraries for NLP, machine learning, and visualization.

---

### 2. Data Loading (UCI Drug Review, ID=461)
Load features and target, then merge into a single DataFrame.

---

### 3. Text Construction
Combine:
- `benefitsReview`
- `sideEffectsReview`
- `commentsReview`

➡️ into `full_review` to capture full user context.

---

### 4. Text Cleaning
- lowercasing
- removing special characters
- removing duplicates & empty rows

➡️ reduces noise and improves feature quality

---

### 5. Sentiment Labeling
Convert ratings into:
- negative
- neutral
- positive

➡️ defines a 3-class classification problem

---

### 6. NLP Analysis (SpaCy: POS / NER)
- tokenization
- POS tagging
- entity recognition

Create `filtered_review` (e.g. ADJ + NOUN) to test linguistic feature filtering.

---

### 7. Train/Test Split + TF-IDF
- split dataset
- vectorize text using TF-IDF

➡️ converts text into numerical features

---

### 8. Handling Class Imbalance
- SMOTE (if available)
- fallback oversampling

➡️ improves learning on minority classes

---

### 9. Model Training & Comparison
Models:
- Logistic Regression
- SVM

Feature variants:
- TF-IDF
- TF-IDF + resampling
- TF-IDF + POS + resampling

---

### 10. Model Evaluation
Metrics:
- accuracy
- precision / recall / F1
- balanced accuracy
- MCC
- Cohen’s kappa

➡️ deeper evaluation beyond accuracy

---

### 11. Model Interpretation
Extract top TF-IDF features for each class.

➡️ understand what drives predictions

---

### 12. Feature Extension
Add `effectiveness` as an additional numeric feature.

➡️ test impact on performance

---

### 13. Advanced Diagnostics
- confusion matrix
- per-class metrics
- most frequent misclassifications

➡️ identify model weaknesses

---

### 14. Streamlit Integration
- stable plotting (`st.pyplot(fig)`)
- simplified 2-class output for usability

➡️ bridges analysis with real-world application

---

### 15. Threshold Calibration (2-Class UI)
- split into train / calibration / test for binary setup
- tune decision threshold on validation probabilities instead of fixed `0.5`
- select threshold by ranking `f1_positive`, `balanced_accuracy`, then `precision_positive`

➡️ improves reliability of final "godny uwagi" vs "raczej nie" decision in Streamlit
