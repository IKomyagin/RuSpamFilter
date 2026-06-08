import tempfile
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login

model_path = "../models/my_ruSpam_model"

login(token="token")

print("Загрузка модели и токенизатора...")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

with tempfile.TemporaryDirectory() as tmp_dir:
    print(f"Сохраняем чистую модель во временную папку {tmp_dir}")
    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)

    repo_id = "AsuUser/ruSpam_model"
    print(f"Загружаем на Hub в репозиторий {repo_id}...")
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

print("Модель загружена на hf!")