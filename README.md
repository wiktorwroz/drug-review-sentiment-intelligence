# Drug Review Sentiment Intelligence

This project analyzes drug reviews with NLP (cleaning, POS/NER, TF-IDF) and compares multiple classification setups, including SVM and logistic regression with resampling variants.
The Streamlit app presents a user-facing 2-class recommendation (`godny uwagi` vs `raczej nie`), where neutral reviews are conservatively grouped with negative to avoid misleading conclusions.

## Main Files

- `DrugSentiment_POS_NER.ipynb` - full notebook pipeline, experiments, and diagnostics
- `drug_sentiment_streamlit.py` - Streamlit app for interactive drug-level recommendations

## Run Streamlit

```bash
python3 -m streamlit run "/Users/turfian/Mastering-NLP-from-Foundations-to-LLMs/drug_sentiment_streamlit.py"
```
