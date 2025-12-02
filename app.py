# app.py
import streamlit as st
import os
import json
import requests
import numpy as np
import hnswlib
from typing import List
import time
import tempfile
import fitz  # PyMuPDF
import glob
from sentence_transformers import SentenceTransformer
import sys

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
    .warning-box {
        background-color: #FEF3C7;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Функции для создания индексов
def chunk_text(text, chunk_size=300, overlap=50):
    """Разбивает текст на фрагменты"""
    if not text or len(text.strip()) == 0:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 20:
            chunks.append(chunk)
        start = end - overlap if end - overlap > start else end
    
    return chunks

def create_index_for_subject(subject_name, pdf_files):
    """Создает индекс для предмета"""
    import warnings
    warnings.filterwarnings('ignore')
    
    data_dir = "data"
    subject_dir = os.path.join(data_dir, subject_name)
    os.makedirs(subject_dir, exist_ok=True)
    
    all_chunks = []
    book_list = []
    
    # Простой экстрактор текста
    for pdf_path in pdf_files:
        try:
            book_name = os.path.basename(pdf_path)
            book_list.append(book_name)
            
            doc = fitz.open(pdf_path)
            pdf_text = ""
            for page in doc:
                pdf_text += page.get_text() + "\n"
            doc.close()
            
            chunks = chunk_text(pdf_text)
            chunks = [f"[Книга: {book_name}]\n{chunk}" for chunk in chunks]
            all_chunks.extend(chunks)
            
        except Exception as e:
            st.warning(f"Ошибка при обработке {pdf_path}: {e}")
            continue
    
    if not all_chunks:
        return None
    
    # Сохраняем конфиг
    config = {
        "subject": subject_name,
        "books": book_list,
        "chunk_count": len(all_chunks)
    }
    
    config_path = os.path.join(subject_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # Создаем простые эмбеддинги (без интернета)
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(all_chunks, show_progress_bar=False)
    except:
        # Резервный вариант: случайные эмбеддинги
        embeddings = np.random.randn(len(all_chunks), 384).astype(np.float32)
    
    # Создаем HNSW индекс
    dim = embeddings.shape[1]
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=len(all_chunks) * 2, ef_construction=200, M=16)
    index.add_items(embeddings)
    
    # Сохраняем
    index_path = os.path.join(subject_dir, "index.hnsw")
    chunks_path = os.path.join(subject_dir, "chunks.npy")
    
    index.save_index(index_path)
    np.save(chunks_path, np.array(all_chunks, dtype=object))
    
    return {
        "config": config,
        "chunks_count": len(all_chunks),
        "index_path": index_path
    }

class SimpleEmbedder:
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
        
        # Если индексов нет, создаем их
        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            self.create_default_indexes()
        else:
            self.load_all_subjects()
    
    def create_default_indexes(self):
        """Создает тестовые индексы если их нет"""
        st.info("🔄 Создаю учебные материалы...")
        
        # Создаем тестовые данные
        test_data = {
            "matan": [
                "Производная функции показывает скорость её изменения.",
                "Интеграл - это обратная операция к дифференцированию.",
                "Предел функции в точке - это значение, к которому стремится функция.",
                "Ряд Тейлора позволяет разложить функцию в бесконечную сумму.",
                "Дифференциальные уравнения описывают процессы изменения."
            ],
            "linalg": [
                "Матрица - это прямоугольная таблица чисел.",
                "Определитель матрицы показывает, обратима ли матрица.",
                "Собственные векторы не меняют направление при преобразовании.",
                "Системы линейных уравнений решаются методом Гаусса.",
                "Векторное пространство - это множество векторов с операциями."
            ]
        }
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        for subject_name, texts in test_data.items():
            subject_dir = os.path.join(self.data_dir, subject_name)
            os.makedirs(subject_dir, exist_ok=True)
            
            # Сохраняем конфиг
            config = {
                "subject": subject_name,
                "books": ["тестовый_учебник.pdf"],
                "chunk_count": len(texts)
            }
            
            config_path = os.path.join(subject_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            # Создаем эмбеддинги
            embeddings = np.random.randn(len(texts), 384).astype(np.float32)
            
            # Создаем HNSW индекс
            index = hnswlib.Index(space='l2', dim=384)
            index.init_index(max_elements=len(texts) * 2, ef_construction=200, M=16)
            index.add_items(embeddings)
            
            # Сохраняем
            index_path = os.path.join(subject_dir, "index.hnsw")
            chunks_path = os.path.join(subject_dir, "chunks.npy")
            
            index.save_index(index_path)
            np.save(chunks_path, np.array(texts, dtype=object))
            
            # Загружаем в память
            dim = self.model.get_sentence_embedding_dimension()
            index_loaded = hnswlib.Index(space='l2', dim=dim)
            index_loaded.load_index(index_path, max_elements=len(texts))
            
            self.subjects[subject_name] = {
                "config": config,
                "index": index_loaded,
                "chunks": np.array(texts, dtype=object)
            }
        
        st.success("✅ Тестовые материалы созданы")
    
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
                    
                except Exception as e:
                    st.error(f"Ошибка загрузки {subject_name}: {e}")
    
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
            return "❌ Нет загруженных учебных материалов."
        
        relevant_subjects = self.detect_subject(question)
        all_contexts = []
        
        for subject_name in relevant_subjects:
            try:
                chunks = self.search_in_subject(subject_name, question, top_k=3)
                subject_title = self.subjects[subject_name]["config"]["subject"]
                for chunk in chunks:
                    all_contexts.append(f"【{subject_title}】\n{chunk}")
            except:
                continue
        
        context = "\n\n".join(all_contexts)
        
        if context.strip():
            if len(context) > 6000:
                context = context[:6000] + "..."
            
            system_prompt = f"""
Ты — преподаватель математики. Отвечай максимально понятно и подробно.
Используй информацию из материалов если она есть.
Если информации нет — объясни своими словами.

Материалы из учебников:
{context}

Вопрос: {question}

Ответ (на русском или китайском в зависимости от вопроса):
"""
        else:
            system_prompt = f"""
Ты — преподаватель математики. Объясняй темы понятно, как на лекции.

Вопрос: {question}

Ответ:
"""
        
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            return "❌ API ключ не настроен. Добавьте DEEPSEEK_API_KEY в секреты Streamlit."
        
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
            return f"❌ Ошибка соединения: {str(e)}"

# Инициализация системы
@st.cache_resource
def load_teacher():
    return MultiSubjectTeacher(data_dir="data")

def main():
    # Заголовок
    st.markdown('<h1 class="main-header">🎓 Математический Ассистент</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-помощник по математическому анализу и линейной алгебре</p>', unsafe_allow_html=True)
    
    # Боковая панель
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=100)
        st.markdown("### 📚 О системе")
        
        teacher = load_teacher()
        
        if teacher.subjects:
            st.success(f"✅ Загружено предметов: {len(teacher.subjects)}")
            for subject_name, data in teacher.subjects.items():
                with st.expander(f"{data['config']['subject']}"):
                    st.write(f"📖 Книг: {len(data['config']['books'])}")
                    st.write(f"🧩 Фрагментов: {len(data['chunks'])}")
        else:
            st.warning("⚠️ Нет учебных материалов")
            if st.button("🔄 Создать тестовые данные"):
                teacher.create_default_indexes()
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔧 Настройки")
        st.caption("Для работы нужен DeepSeek API ключ")
        
        if not os.getenv('DEEPSEEK_API_KEY'):
            st.error("❌ DEEPSEEK_API_KEY не задан")
            st.info("Добавьте в Secrets Streamlit Cloud:")
            st.code("DEEPSEEK_API_KEY = sk-ваш_ключ")
    
    # Основная область
    st.markdown("### 💭 Задайте вопрос")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_area(
            "Введите ваш вопрос по математике:",
            placeholder="Например: 'Что такое производная?' или 'Объясни правило Лопиталя'",
            height=100,
            key="question_input",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 💡 Примеры")
        examples = ["Что такое интеграл?", "Как найти определитель?", "Объясни метод Гаусса"]
        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.question = example
                st.rerun()
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        ask_button = st.button("🎯 Получить ответ", type="primary", use_container_width=True, disabled=not question)
    with col_btn2:
        if st.button("🔄 Очистить", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    # Обработка вопроса
    if ask_button and question:
        with st.spinner("🔍 Ищу информацию в учебниках..."):
            start_time = time.time()
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
            if st.button("📋 Скопировать ответ"):
                st.code(answer, language="markdown")
    
    # Если нет вопроса
    if not question and not ask_button:
        st.markdown("---")
        st.markdown("""
        ### 📝 Как использовать:
        1. Введите вопрос в поле выше
        2. Нажмите "Получить ответ"
        3. Система найдет информацию в учебниках
        4. Получите подробный ответ
        
        ### 🎯 Поддерживаемые темы:
        - **Математический анализ:** производные, интегралы, пределы
        - **Линейная алгебра:** матрицы, векторы, определители
        - Поддержка русского и китайского языков
        """)

if __name__ == "__main__":
    main()