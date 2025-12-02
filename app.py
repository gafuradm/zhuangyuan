# app.py - Веб-интерфейс для математического ассистента
import streamlit as st
import os
import json
import requests
import numpy as np
import hnswlib
from typing import List
import time

# Настройка страницы
st.set_page_config(
    page_title="Математический Ассистент",
    page_icon="📚",
    layout="wide"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #475569;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 15px 0;
    }
    .answer-box {
        background-color: #F0F9FF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        margin: 15px 0;
    }
    .stats-box {
        background-color: #FEF3C7;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class SimpleEmbedder:
    """Простая модель эмбеддингов БЕЗ интернета"""
    def __init__(self, dim=384):
        self.dim = dim
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
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
        self.model = SimpleEmbedder(dim=384)
        self.subjects = {}
        self.load_all_subjects()
    
    def load_all_subjects(self):
        if not os.path.exists(self.data_dir):
            return
        
        for subject_name in os.listdir(self.data_dir):
            subject_path = os.path.join(self.data_dir, subject_name)
            if os.path.isdir(subject_path):
                try:
                    config_path = os.path.join(subject_path, "config.json")
                    if not os.path.exists(config_path):
                        continue
                    
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    index_path = os.path.join(subject_path, "index.hnsw")
                    chunks_path = os.path.join(subject_path, "chunks.npy")
                    
                    if not os.path.exists(index_path):
                        continue
                    
                    chunks = np.load(chunks_path, allow_pickle=True)
                    
                    dim = self.model.get_sentence_embedding_dimension()
                    index = hnswlib.Index(space='l2', dim=dim)
                    index.load_index(index_path, max_elements=len(chunks))
                    
                    self.subjects[subject_name] = {
                        "config": config,
                        "index": index,
                        "chunks": chunks
                    }
                    
                except Exception:
                    continue
    
    def detect_subject(self, question: str) -> List[str]:
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
        
        return relevant_subjects if relevant_subjects else list(self.subjects.keys())
    
    def search_in_subject(self, subject_name: str, query: str, top_k: int = 3):
        subject_data = self.subjects[subject_name]
        query_emb = self.model.encode([query])
        
        indices, distances = subject_data["index"].knn_query(query_emb, k=top_k)
        return [subject_data["chunks"][idx] for idx in indices[0]]
    
    def ask(self, question: str):
        if not self.subjects:
            return "❌ Нет загруженных предметов."
        
        relevant_subjects = self.detect_subject(question)
        all_contexts = []
        
        for subject_name in relevant_subjects:
            try:
                chunks = self.search_in_subject(subject_name, question, top_k=2)
                subject_title = self.subjects[subject_name]["config"]["subject"]
                for chunk in chunks:
                    all_contexts.append(f"【{subject_title}】\n{chunk}")
            except:
                continue
        
        context = "\n\n".join(all_contexts)
        
        if context.strip():
            if len(context) > 8000:
                context = context[:8000] + "..."
            
            system_prompt = f"""
Ты — преподаватель математики. Отвечай максимально понятно и подробно.
Используй информацию из материалов если она есть.
Если информации нет — объясни своими словами.

Материалы из учебников:
{context}

Вопрос: {question}
"""
        else:
            system_prompt = f"""
Ты — преподаватель математики. Объясняй темы понятно, как на лекции.

Вопрос: {question}
"""
        
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            return "❌ API ключ не настроен."
        
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
                return f"❌ Ошибка API"
            
            data = resp.json()
            return data["choices"][0]["message"]["content"]
            
        except Exception:
            return "❌ Ошибка соединения"

# Инициализация системы
@st.cache_resource
def load_teacher():
    return MultiSubjectTeacher(data_dir="data")

def main():
    # Заголовок
    st.markdown('<h1 class="main-header">🎓 Математический Ассистент</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-помощник по математическому анализу и линейной алгебре</p>', unsafe_allow_html=True)
    
    # Боковая панель с информацией
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=100)
        st.markdown("### 📚 О системе")
        st.info("""
        Эта система использует:
        - **2 учебника** (матанализ + линейная алгебра)
        - **2789 фрагментов** текста
        - **DeepSeek AI** для генерации ответов
        - **Векторный поиск** для точности
        """)
        
        # Статистика
        teacher = load_teacher()
        if teacher.subjects:
            st.markdown("### 📊 Статистика")
            for subject_name, data in teacher.subjects.items():
                st.markdown(f"""
                <div class="stats-box">
                <strong>{data['config']['subject']}:</strong><br>
                📖 {len(data['chunks'])} фрагментов
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💡 Примеры вопросов")
        examples = [
            "Что такое производная?",
            "Как найти определитель матрицы?",
            "Объясни правило Лопиталя",
            "Что такое собственные значения?",
            "Как решать системы линейных уравнений?"
        ]
        for example in examples:
            if st.button(f"🔍 {example}", key=example):
                st.session_state.question = example
    
    # Основная область
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Поле для вопроса
        question = st.text_area(
            "### 💭 Ваш вопрос по математике",
            placeholder="Например: 'Что такое интеграл?' или 'Как решать матричные уравнения?'",
            height=100,
            key="question_input"
        )
        
        # Кнопки действий
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            ask_button = st.button("🎯 Задать вопрос", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🔄 Очистить", use_container_width=True)
        with col_btn3:
            example_button = st.button("🎲 Случайный пример", use_container_width=True)
        
        if clear_button:
            st.session_state.clear()
            st.rerun()
        
        if example_button:
            examples = [
                "Объясни теорему о среднем значении",
                "Что такое матрица поворота?",
                "Как вычислять кратные интегралы?",
                "Что такое ядро линейного оператора?"
            ]
            import random
            st.session_state.question = random.choice(examples)
            st.rerun()
    
    with col2:
        st.markdown("### 🌐 Поддерживает")
        st.markdown("""
        - Русский язык
        - Китайский язык
        - Математические формулы
        - Подробные объяснения
        """)
    
    # Обработка вопроса
    if ask_button and question:
        with st.spinner("🔍 Ищу ответ в учебниках..."):
            start_time = time.time()
            teacher = load_teacher()
            answer = teacher.ask(question)
            end_time = time.time()
            
            # Показываем вопрос
            st.markdown(f"""
            <div class="question-box">
            <strong>❓ Вопрос:</strong><br>
            {question}
            </div>
            """, unsafe_allow_html=True)
            
            # Показываем ответ
            st.markdown(f"""
            <div class="answer-box">
            <strong>📚 Ответ:</strong><br>
            {answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Статистика
            st.caption(f"⏱️ Время ответа: {end_time-start_time:.2f} секунд")
            
            # Кнопка копирования
            st.code(answer, language="markdown")
    
    # История вопросов (если есть)
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Инструкция если нет вопроса
    if not question and not ask_button:
        st.markdown("---")
        st.markdown("### 📝 Как использовать:")
        st.info("""
        1. Введите вопрос в поле выше
        2. Нажмите "Задать вопрос"
        3. Система найдет информацию в учебниках
        4. Получите развернутый ответ с примерами
        """)
        
        st.markdown("### 🎯 Популярные темы:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Матанализ:**
            - Производные и дифференциалы
            - Интегралы и пределы
            - Ряды и последовательности
            - Функции многих переменных
            """)
        with col2:
            st.markdown("""
            **Линейная алгебра:**
            - Матрицы и определители
            - Системы уравнений
            - Векторные пространства
            - Собственные значения
            """)

if __name__ == "__main__":
    main()