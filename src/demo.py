import torch
import gradio as gr

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

genres = ['Kinh tế', 'Giáo dục', 'Xe', 'Sức khoẻ', 'Công nghệ - Game']

model_name = "mob2711/phoBERT_finetune_news_classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

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
    examples=["Chủ tịch HĐQT Trường quốc tế AISVN đề xuất hỗ trợ 125 tỉ đồng", 
              "Chuyên gia tài chính Nguyễn Trí Hiếu bị 'hack' gần 500 triệu đồng, ngân hàng im lặng suốt 3 tháng?",
              "'Siêu nhân bảo vệ nụ cười' P/S xuất hiện tại 35 Siêu thị Co.opmart trên cả nước",
              "Microsoft hợp tác OpenAI phát triển siêu máy tính AI giá hơn 100 tỉ USD",
              "Triệu hồi 170.000 xe điện Hyundai, Kia bị lỗi mất điện",
              ]
)

demo.launch()