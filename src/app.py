import gradio as gr
from inference import predict_spam

def classify_text(text):
    label, prob = predict_spam(text)   # label уже "СПАМ" или "НЕ СПАМ"
    return f"{label} (вероятность спама: {prob:.4f})"

iface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(label="Введите сообщение"),
    outputs=gr.Textbox(label="Результат"),
    title="Детектор спама (BERT)",
    description="Модель на основе ruBERT (дообучена на датасете ruSpam)",
    examples=[
        ["Привет! Как дела?"],
        ["Акция! Только сегодня! Скидка 90%! Переходи по ссылке: http://spam.ru"],
        ["Вам начислен выигрыш. Зайдите в личный кабинет"],
        ["Здравствуйте, назначим встречу на завтра?"]
    ],
)
iface.launch()