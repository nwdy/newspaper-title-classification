import torch
import gradio as gr

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

genres = ['Kinh tế', 'Giáo dục', 'Xe', 'Sức khoẻ', 'Công nghệ - Game']

tokenizer = AutoTokenizer.from_pretrained("mob2711/phoBERT_finetune_news_classification")
model = AutoModelForSequenceClassification.from_pretrained("mob2711/phoBERT_finetune_news_classification")

def tokenize(text):
    encoded_text = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    return encoded_text

def predict_proba(text_data):
    encoded_data = tokenize(text_data)
    with torch.no_grad():
        outputs = model(**encoded_data)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)[0]
    label_probs = {genres[id]: prob for id, prob in enumerate(probabilities)}
    return label_probs
    
# Interface
input_text = gr.Textbox(lines=2, label="Enter the title")
output_text = gr.Label(label="Predicted Probabilities")

demo = gr.Interface(
    fn=predict_proba, 
    inputs=input_text, 
    outputs=output_text, 
    title="Newspaper Title Classifier",
)

demo.launch()