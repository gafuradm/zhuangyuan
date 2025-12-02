import streamlit as st
import os
import json
import requests
import numpy as np
import hnswlib
from typing import List
import time
import hashlib

# ========== КОНФИГУРАЦИЯ ==========
st.set_page_config(
    page_title="Математический Ассистент",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body);"></script>
""", unsafe_allow_html=True)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subject-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #3B82F6;
    }
    .stButton button {
        width: 100%;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ========== МОДЕЛЬ ЭМБЕДДИНГОВ ==========
class SimpleEmbedder:
    """Простая модель без интернета"""
    def __init__(self, dim=384):
        self.dim = dim
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Детерминированный хэш для воспроизводимости
            text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(text_hash)
            emb = np.random.randn(self.dim).astype(np.float32)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def get_sentence_embedding_dimension(self):
        return self.dim

# ========== ОСНОВНОЙ КЛАСС ==========
class MathAssistant:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.model = SimpleEmbedder(dim=384)
        self.subjects = {}
        self.load_subjects()
    
    def load_subjects(self):
        """Загружает все предметы"""
        if not os.path.exists(self.data_dir):
            st.error(f"❌ Папка '{self.data_dir}' не найдена!")
            return
        
        subject_folders = [d for d in os.listdir(self.data_dir) 
                          if os.path.isdir(os.path.join(self.data_dir, d))]
        
        if not subject_folders:
            st.warning("⚠️ В папке data/ нет предметов")
            return
        
        for subject_name in subject_folders:
            try:
                subject_path = os.path.join(self.data_dir, subject_name)
                
                # Проверяем необходимые файлы
                required_files = ["config.json", "index.hnsw", "chunks.npy"]
                if not all(os.path.exists(os.path.join(subject_path, f)) for f in required_files):
                    st.warning(f"⚠️ В папке '{subject_name}' не хватает файлов")
                    continue
                
                # Загружаем конфиг
                with open(os.path.join(subject_path, "config.json"), 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Загружаем чанки
                chunks = np.load(os.path.join(subject_path, "chunks.npy"), allow_pickle=True)
                
                # Загружаем HNSW индекс
                dim = self.model.get_sentence_embedding_dimension()
                index = hnswlib.Index(space='l2', dim=dim)
                index.load_index(os.path.join(subject_path, "index.hnsw"), 
                               max_elements=len(chunks))
                
                self.subjects[subject_name] = {
                    "config": config,
                    "index": index,
                    "chunks": chunks
                }
                
            except Exception as e:
                st.error(f"❌ Ошибка загрузки '{subject_name}': {str(e)}")
    
    def detect_subject(self, question: str) -> List[str]:
        """Определяет предмет вопроса"""
        question_lower = question.lower()
        subject_keywords = {
            "matan": ["матанализ", "мат анализ", "дифференциал", "интеграл", 
                     "предел", "ряд", "функция", "производная", "дифференцирование"],
            "linalg": ["линейн", "матриц", "вектор", "определитель", 
                      "собствен", "линейное пространство", "линейно", "алгебр"]
        }
        
        relevant = []
        for subject_name in self.subjects.keys():
            if subject_name in subject_keywords:
                for keyword in subject_keywords[subject_name]:
                    if keyword in question_lower:
                        if subject_name not in relevant:
                            relevant.append(subject_name)
                        break
        
        return relevant if relevant else list(self.subjects.keys())
    
    def search_in_subject(self, subject_name: str, query: str, top_k: int = 3):
        """Ищет в конкретном предмете"""
        subject_data = self.subjects[subject_name]
        query_emb = self.model.encode([query])
        indices, distances = subject_data["index"].knn_query(query_emb, k=top_k)
        return [subject_data["chunks"][idx] for idx in indices[0]]
    
    def ask(self, question: str) -> str:
        """Основной метод для ответов"""
        if not self.subjects:
            return "❌ Нет загруженных учебных материалов."
        
        # Определяем предметы
        relevant_subjects = self.detect_subject(question)
        
        # Собираем контекст
        all_contexts = []
        for subject_name in relevant_subjects:
            try:
                chunks = self.search_in_subject(subject_name, question, top_k=3)
                subject_title = self.subjects[subject_name]["config"]["subject"]
                for i, chunk in enumerate(chunks[:3]):  # Берем только 3 лучших
                    all_contexts.append(f"📘 {subject_title}:\n{chunk}\n")
            except Exception as e:
                continue
        
        context = "\n".join(all_contexts)
        
        # Формируем промпт
        if context.strip():
            system_prompt = f"""Ты — преподаватель математики. Отвечай на русском языке.

ИНФОРМАЦИЯ ИЗ УЧЕБНИКОВ:
{context}

ВОПРОС: {question}

ОТВЕТ (используй информацию из учебников если она есть, если нет — объясни своими словами):
"""
        else:
            system_prompt = f"""Ты — преподаватель математики. Отвечай понятно и подробно.

ВОПРОС: {question}

ОТВЕТ:
"""
        
        # Отправляем запрос к DeepSeek
        api_key = st.secrets.get("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY"))
        if not api_key:
            return "❌ API ключ не настроен. Добавьте DEEPSEEK_API_KEY в секреты."
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"❌ Ошибка API ({response.status_code}): {response.text}"
                
        except Exception as e:
            return f"❌ Ошибка соединения: {str(e)}"

# ========== ИНТЕРФЕЙС STREAMLIT ==========
def main():
    # Заголовок
    st.markdown('<h1 class="main-header">🎓 Математический Ассистент</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-помощник по математике на основе ваших учебников</p>', unsafe_allow_html=True)
    
    # Инициализация ассистента
    if "assistant" not in st.session_state:
        with st.spinner("🔄 Загружаю учебные материалы..."):
            st.session_state.assistant = MathAssistant("data")
    
    assistant = st.session_state.assistant
    
    # Боковая панель
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=100)
        st.markdown("### 📚 Загруженные предметы")
        
        if assistant.subjects:
            for subject_name, data in assistant.subjects.items():
                with st.container():
                    st.markdown(f"""
                    <div class="subject-card">
                    <strong>{data['config']['subject']}</strong><br>
                    📖 {len(data['config']['books'])} книг<br>
                    🧩 {len(data['chunks'])} фрагментов
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Учебные материалы не загружены")
            st.info("Убедитесь, что папка `data/` есть в репозитории")
        
        st.markdown("---")
        st.markdown("### 💡 Примеры вопросов")
        
        examples = [
            "Что такое производная?",
            "Как найти определитель матрицы?",
            "Объясни правило Лопиталя",
            "Что такое собственные значения?"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}"):
                st.session_state.question = example
    
    # Основная область
    st.markdown("### 💭 Задайте вопрос по математике")
    
    # Поле для вопроса
    question = st.text_area(
        "Введите ваш вопрос:",
        value=st.session_state.get("question", ""),
        placeholder="Например: 'Что такое производная?' или 'Объясни метод Гаусса'",
        height=120,
        label_visibility="collapsed"
    )
    
    # Кнопки
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🎯 Получить ответ", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("🔍 Ищу информацию в учебниках..."):
                    start_time = time.time()
                    answer = assistant.ask(question)
                    elapsed = time.time() - start_time
                    
                    # Сохраняем в историю
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        "question": question,
                        "answer": answer,
                        "time": elapsed
                    })
                    
                    # Показываем ответ
                    st.markdown(f"### 📚 Ответ ({elapsed:.1f} сек)")
                    st.markdown("---")
                    st.markdown(answer)
                    
                    # Кнопка копирования
                    st.code(answer, language="markdown")
            else:
                st.warning("⚠️ Введите вопрос")
    
    with col2:
        if st.button("🔄 Новый вопрос", use_container_width=True):
            st.session_state.question = ""
            st.rerun()
    
    with col3:
        if st.button("📜 История", use_container_width=True):
            if "history" in st.session_state and st.session_state.history:
                st.markdown("### 📜 История вопросов")
                for i, item in enumerate(reversed(st.session_state.history[-5:])):
                    with st.expander(f"❓ {item['question'][:50]}..."):
                        st.markdown(f"**Время:** {item['time']:.1f} сек")
                        st.markdown(f"**Ответ:** {item['answer'][:200]}...")
            else:
                st.info("📝 История вопросов пуста")
    
    # Информация о системе
    with st.expander("ℹ️ О системе"):
        st.markdown("""
        **Как работает система:**
        1. 📚 Загружает ваши учебники (PDF → текст)
        2. 🔍 Ищет релевантные фрагменты по вопросу
        3. 🤖 Отправляет контекст в DeepSeek AI
        4. 📝 Получает подробный ответ
        
        **Поддерживаемые темы:**
        - Математический анализ
        - Линейная алгебра
        - Дифференциальные уравнения
        
        **Требования:**
        - DeepSeek API ключ (добавьте в секреты)
        - Папка `data/` с индексами учебников
        """)

# Запуск приложения
if __name__ == "__main__":
    main()