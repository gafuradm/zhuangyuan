import os
import json
import requests
import numpy as np
import hnswlib
from typing import List

class SimpleEmbedder:
    """Простая модель эмбеддингов БЕЗ интернета"""
    def __init__(self, dim=384):
        self.dim = dim
        print(f"✅ Использую простую модель ({dim}D)")
    
    def encode(self, texts):
        """Создаем псевдо-эмбеддинги на основе текста"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Простая хэш-функция для создания детерминированных эмбеддингов
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            emb = np.random.randn(self.dim).astype(np.float32)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def get_sentence_embedding_dimension(self):
        return self.dim

class MultiSubjectTeacher:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.model = SimpleEmbedder(dim=384)  # Используем простую модель
        self.subjects = {}
        self.load_all_subjects()
    
    def load_all_subjects(self):
        """Загружаем все предметы из data_dir"""
        if not os.path.exists(self.data_dir):
            print(f"❌ Директория {self.data_dir} не найдена!")
            return
        
        print("📚 Загружаю предметы...")
        for subject_name in os.listdir(self.data_dir):
            subject_path = os.path.join(self.data_dir, subject_name)
            if os.path.isdir(subject_path):
                try:
                    # Загружаем конфиг
                    config_path = os.path.join(subject_path, "config.json")
                    if not os.path.exists(config_path):
                        print(f"  ⚠️  Нет config.json в {subject_name}, пропускаю...")
                        continue
                    
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    # Загружаем HNSW индекс и чанки
                    index_path = os.path.join(subject_path, "index.hnsw")
                    chunks_path = os.path.join(subject_path, "chunks.npy")
                    
                    if not os.path.exists(index_path):
                        print(f"  ⚠️  Нет index.hnsw в {subject_name}, пропускаю...")
                        continue
                    
                    chunks = np.load(chunks_path, allow_pickle=True)
                    
                    # Загружаем HNSW индекс
                    dim = self.model.get_sentence_embedding_dimension()
                    index = hnswlib.Index(space='l2', dim=dim)
                    index.load_index(index_path, max_elements=len(chunks))
                    
                    self.subjects[subject_name] = {
                        "config": config,
                        "index": index,
                        "chunks": chunks
                    }
                    print(f"  ✅ {config['subject']} ({subject_name}) - {len(chunks)} фрагментов")
                    
                except Exception as e:
                    print(f"  ❌ Ошибка загрузки {subject_name}: {e}")
        
        print(f"🎯 Всего загружено предметов: {len(self.subjects)}")
    
    def detect_subject(self, question: str) -> List[str]:
        """Определяем, к каким предметам относится вопрос"""
        question_lower = question.lower()
        subject_keywords = {
            "matan": ["матанализ", "мат анализ", "дифференциал", "интеграл", 
                     "предел", "ряд", "функция", "производная", "дифференцирование"],
            "linalg": ["линейн", "матриц", "вектор", "определитель", 
                      "собствен", "линейное пространство", "линейно", "алгебр"]
        }
        
        relevant_subjects = []
        for subject_name in self.subjects.keys():
            if subject_name in subject_keywords:
                for keyword in subject_keywords[subject_name]:
                    if keyword in question_lower:
                        if subject_name not in relevant_subjects:
                            relevant_subjects.append(subject_name)
                        break
        
        # Если не нашли, используем все предметы
        return relevant_subjects if relevant_subjects else list(self.subjects.keys())
    
    def search_in_subject(self, subject_name: str, query: str, top_k: int = 3):
        """Ищем в конкретном предмете"""
        subject_data = self.subjects[subject_name]
        query_emb = self.model.encode([query])
        
        # HNSWlib поиск
        indices, distances = subject_data["index"].knn_query(query_emb, k=top_k)
        return [subject_data["chunks"][idx] for idx in indices[0]]
    
    def ask(self, question: str):
        if not self.subjects:
            return "❌ Нет загруженных предметов. Сначала создайте индексы."
        
        # Определяем предметы
        relevant_subjects = self.detect_subject(question)
        print(f"🔍 Ищу в предметах: {', '.join(relevant_subjects)}")
        
        # Собираем контекст из всех релевантных предметов
        all_contexts = []
        for subject_name in relevant_subjects:
            try:
                chunks = self.search_in_subject(subject_name, question, top_k=5)
                subject_title = self.subjects[subject_name]["config"]["subject"]
                for chunk in chunks:
                    all_contexts.append(f"【{subject_title}】\n{chunk}")
            except:
                continue
        
        context = "\n\n".join(all_contexts)
        
        if context.strip():
            # Ограничиваем контекст чтобы не превысить лимит токенов
            if len(context) > 8000:
                context = context[:8000] + "..."
            
            system_prompt = f"""
Ты — универсальный преподаватель математики.
Используй ТОЛЬКО информацию из предоставленных материалов.
Если в материалах нет ответа — скажи об этом и предложи объяснить своими словами.

Материалы:
{context}
"""
        else:
            system_prompt = "Ты — преподаватель математики. Объясняй темы понятно, как на лекции."
        
        # Отправляем запрос к DeepSeek
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            return "❌ Ошибка: Не задан DEEPSEEK_API_KEY"
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "max_tokens": 2000,
            "temperature": 0.7
        }
        
        try:
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=120
            )
            
            if resp.status_code != 200:
                return f"❌ Ошибка API: {resp.status_code}"
            
            data = resp.json()
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            return f"❌ Ошибка соединения: {e}"

# ---- ИСПОЛЬЗОВАНИЕ ----
if __name__ == "__main__":
    # Проверяем API ключ
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ Ошибка: DEEPSEEK_API_KEY не задан!")
        print("   Экспортируйте переменную: export DEEPSEEK_API_KEY='ваш_ключ'")
        exit(1)
    
    teacher = MultiSubjectTeacher(data_dir="data")
    
    if not teacher.subjects:
        print("❌ Нет загруженных предметов.")
        print("   Используйте embed_local.py для создания индексов:")
        print("   python embed_local.py --subject matan --pdf-dir pdfs/matan")
        print("   python embed_local.py --subject linalg --pdf-dir pdfs/linalg")
        exit(1)
    
    print("\n" + "="*50)
    print("🎓 МАТЕМАТИЧЕСКИЙ АССИСТЕНТ (ОФФЛАЙН РЕЖИМ)")
    print("="*50)
    print("Доступные предметы:")
    for subject_name, data in teacher.subjects.items():
        print(f"  • {data['config']['subject']}: {len(data['chunks'])} фрагментов")
    print("="*50)
    print("📝 Примеры вопросов:")
    print("  • 'Что такое производная?'")
    print("  • 'Объясни правило Лопиталя'")
    print("  • 'Что такое определитель матрицы?'")
    print("  • 'Как решать системы линейных уравнений?'")
    print("="*50)
    
    print("\n✅ Система готова к работе! Задавайте вопросы.")
    
    while True:
        q = input("\n🎯 Ваш вопрос (или 'exit'): ").strip()
        if q.lower() == 'exit':
            print("👋 До свидания!")
            break
        if not q:
            continue
        
        print("⏳ Ищу ответ...")
        answer = teacher.ask(q)
        print("\n" + "📚 ОТВЕТ:")
        print("-" * 60)
        print(answer)
        print("-" * 60)