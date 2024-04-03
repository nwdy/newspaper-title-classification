from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import torch.nn.functional as F

class PhoBERTPredictor:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def tokenize(self, text):
        encoded_text = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        return encoded_text
    
    def predict(self, text_data):
        encoded_data = self.tokenize(text_data)
        with torch.no_grad():
            outputs = self.model(**encoded_data)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        return predictions.cpu().numpy()
    
    def predict_prob(self, text_data):
        encoded_data = self.tokenize(text_data)
        with torch.no_grad():
            outputs = self.model(**encoded_data)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        return predictions.cpu().numpy(), probabilities.cpu().numpy()

def main():
    # Usage
    predictor = PhoBERTPredictor("mob2711/phoBERT_finetune_news_classification")

    # Load test data
    test_data = pd.read_csv('data/test_dataset.csv')

    # Make predictions
    text = test_data['title'].iloc[0]
    predictions, label_probs = predictor.predict_prob(text)

    for idx, (pred, probs) in enumerate(zip(predictions, label_probs)):
        print(f"Text: {test_data['title'][idx]}")
        print(f"Predicted label: {pred}")
        print("Probabilities:")
        for label, prob in enumerate(probs):
            print(f"Label {label}: {prob:.4f}")
        print("=" * 50)

if __name__ == '__main__':
    main()