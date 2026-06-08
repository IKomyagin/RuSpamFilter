import gradio as gr
from inference import predict_spam

def classify_text(text):
    pred, prob = predict_spam(text)
    label = "СПАМ" if pred == 1 else "НЕ СПАМ"
    return f"{label} (вероятность спама: {prob:.4f})"

iface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(label="Введите сообщение"),
    outputs=gr.Textbox(label="Результат"),
    title="Детектор спама (BERT)",
    description="Модель на основе ruBERT (дообучена на датасете ruSpam)"
)
iface.launch()