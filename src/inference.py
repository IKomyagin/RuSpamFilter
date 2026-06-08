from transformers import pipeline

# Для локальной моделим
# classifier = pipeline("text-classification", model="../models/my_ruSpam_model")

# Загрузка модели с Hugging Face Hub
classifier = pipeline("text-classification", model="AsuUser/ruSpam_model")

# Маппинг меток для бинарной классификации (0 — не спам, 1 — спам)
LABELS = {
    "LABEL_0": "НЕ СПАМ",
    "LABEL_1": "СПАМ"
}

def predict_spam(text):
    result = classifier(text)[0]
    label = result["label"]       # Например, "LABEL_1"
    score = result["score"]       # Уверенность модели в этом классе
    return LABELS.get(label, label), score

if __name__ == "__main__":
    test_texts = [
        "Привет! Как дела?",
        "Акция! Только сегодня! Скидка 90%! Переходи по ссылке: http://spam.ru",
        "Вам начислен выигрыш. Зайдите в личный кабинет",
        "Здравствуйте, назначим встречу на завтра?"
    ]
    for t in test_texts:
        label, conf = predict_spam(t)
        print(f"Текст: {t}\nРезультат: {label} (уверенность: {conf:.4f})\n")