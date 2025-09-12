import spacy
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import numpy

app = FastAPI(title="Comment Classfier", version="1.0")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def fast_lemmatize(texts):
    lemmatized = []
    for doc in nlp.pipe(texts, batch_size=1000):
        lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        lemmatized.append(" ".join(lemmas))
    return lemmatized


path = Path(r"F:/Projects/Machine and Deep Learning/Depression_Severity/Final_Modelling/Roberta_bert").resolve()

tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)

model.to(device)
model.eval()

class InputText(BaseModel):
    text: str
    threshold: float = 0.4

@app.get("/")
def root():
    print("Welcome to Comment CLassifier")
        

@app.post("/predict")
def predict(request: InputText):
    lemmas = fast_lemmatize([request.text])

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
    probs_array = numpy.array(probs)
    preds = (probs_array > request.threshold).astype(int).tolist()
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    label_probs = dict(zip(labels, probs))
    label_preds = dict(zip(labels, preds))



    return {
        "original": request.text,
        "lemmas": lemmas[0],
        "probs": label_probs,
        "preds": label_preds
    }