import spacy
import torch
import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def fast_lemmatize(texts):
    lemmatized = []
    for doc in nlp.pipe(texts, batch_size=1000):
        lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        lemmatized.append(" ".join(lemmas))
    return lemmatized

# Load model and tokenizer
path = Path(r"F:/Projects/Machine and Deep Learning/Depression_Severity/Final_Modelling/Roberta_bert").resolve()

tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    str(path),
    local_files_only=True,
    torch_dtype=torch.float32
).to(device)

model.to(device)
model.eval()

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Streamlit UI
st.set_page_config(page_title="Comment Classifier", page_icon="💬", layout="centered")

st.title("💬 Comment Classifier")
st.write("Classify comments into multiple toxicity categories using a fine-tuned RoBERTa model.")
st.write(f"Using device: {device}")
# Input
text_input = st.text_area("Enter a comment:", "")
threshold = st.slider("Prediction threshold", 0.0, 1.0, 0.4, 0.05)

if st.button("Classify"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Lemmatization
        lemmas = fast_lemmatize([text_input])

        # Tokenization
        encoding = tokenizer(
            lemmas,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in encoding.items()})
            probs = torch.sigmoid(outputs.logits)

        probs = probs.cpu().numpy().tolist()[0]
        probs_array = np.array(probs)
        preds = (probs_array > threshold).astype(int).tolist()

        label_probs = dict(zip(labels, probs))
        label_preds = dict(zip(labels, preds))

        st.subheader("Results")
        st.write("**Original Text:**", text_input)
        st.write("**Lemmatized:**", lemmas[0])

        st.write("### Prediction Probabilities")
        for lbl, prob in label_probs.items():
            st.progress(prob)
            st.write(f"**{lbl}**: {prob:.4f}")

        st.write("### Final Predictions")
        st.json(label_preds)
