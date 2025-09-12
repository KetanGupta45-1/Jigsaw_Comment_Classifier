import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

path = r"F:/Projects/Machine and Deep Learning/Depression_Severity/Final_Modelling/Roberta_bert"

# Load the model (force CPU, ensure weights are loaded)
model = AutoModelForSequenceClassification.from_pretrained(
    path,
    local_files_only=True,
    device_map="cpu",
    torch_dtype=torch.float32
)

tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

model.save_pretrained(path, safe_serialization=False)
tokenizer.save_pretrained(path)

print("✅ Model re-saved with pytorch_model.bin")
