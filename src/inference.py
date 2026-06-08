import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Путь к лучшей модели
MODEL_PATH = "../models/my_ruSpam_model"

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Устройство (MPS для Mac, CPU fallback)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

def predict_spam(text: str):
    """
    Принимает один текст (строку).
    Возвращает (label, confidence), где label: 0 - не спам, 1 - спам,
    confidence - вероятность класса "спам" (от 0 до 1).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        spam_prob = probs[0, 1].item()
    return pred, spam_prob

if __name__ == "__main__":
    # Примеры
    examples = [
        "Привет! Как дела?",
        "Акция! Только сегодня! Скидка 90%! Переходи по ссылке: http://spam.ru",
        "Вам начислен выигрыш. Зайдите в личный кабинет",
        "Здравствуйте, назначим встречу на завтра?"
    ]
    for ex in examples:
        pred, prob = predict_spam(ex)
        print(f"Текст: {ex}\nРезультат: {'СПАМ' if pred == 1 else 'НЕ СПАМ'} (вероятность спама: {prob:.4f})\n")