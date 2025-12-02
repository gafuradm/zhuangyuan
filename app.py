import streamlit as st
import os
import json
import requests
import numpy as np
import hnswlib
from typing import List, Dict, Any
import time
import hashlib
import functools

# ========== КОНФИГУРАЦИЯ ==========
st.set_page_config(
    page_title="Математический Ассистент",
    page_icon="📚",
    layout="wide"
)

# Загружаем KaTeX
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<script>
    document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\(', right: '\\)', display: false},
                {left: '\\[', right: '\\]', display: true}
            ],
            throwOnError: false
        });
    });
</script>
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
    .math-content {
        font-size: 1.1em;
        line-height: 1.8;
        margin: 1em 0;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
    }
    .progress-container {
        padding: 15px;
        background: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .quick-query-btn {
        margin: 3px !important;
        font-size: 0.9em !important;
    }
    .stButton > button {
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ========== КЭШИРОВАНИЕ ==========
@st.cache_resource
def get_assistant():
    """Кэшированный загрузчик ассистента"""
    return MathAssistant("data")

@st.cache_data(ttl=300)  # Кэш на 5 минут
def cached_search(_assistant, subject_name: str, query: str, top_k: int = 3):
    """Кэшированный поиск"""
    return _assistant.search_in_subject(subject_name, query, top_k)

# ========== МОДЕЛЬ ЭМБЕДДИНГОВ ==========
class SimpleEmbedder:
    """Оптимизированная модель эмбеддингов"""
    def __init__(self, dim=384):
        self.dim = dim
        self._cache = {}  # Простой кэш
        
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Проверяем кэш
            if text in self._cache:
                embeddings.append(self._cache[text])
                continue
                
            text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(text_hash)
            emb = np.random.randn(self.dim).astype(np.float32)
            # Нормализуем
            emb = emb / np.linalg.norm(emb)
            self._cache[text] = emb
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def get_sentence_embedding_dimension(self):
        return self.dim

# ========== ОСНОВНОЙ КЛАСС (ОПТИМИЗИРОВАННЫЙ) ==========
class MathAssistant:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.model = SimpleEmbedder(dim=384)
        self.subjects = {}
        self.load_subjects()
    
    def load_subjects(self):
        """Быстрая загрузка предметов"""
        if not os.path.exists(self.data_dir):
            return
        
        subject_folders = [d for d in os.listdir(self.data_dir) 
                          if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for subject_name in subject_folders[:3]:  # Ограничиваем количество
            try:
                subject_path = os.path.join(self.data_dir, subject_name)
                
                # Проверяем только необходимые файлы
                config_file = os.path.join(subject_path, "config.json")
                index_file = os.path.join(subject_path, "index.hnsw")
                chunks_file = os.path.join(subject_path, "chunks.npy")
                
                if not all(os.path.exists(f) for f in [config_file, index_file, chunks_file]):
                    continue
                
                # Параллельная загрузка
                import concurrent.futures
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    future_config = executor.submit(self._load_config, config_file)
                    future_chunks = executor.submit(np.load, chunks_file, allow_pickle=True)
                    
                    config = future_config.result()
                    chunks = future_chunks.result()
                    
                    dim = self.model.get_sentence_embedding_dimension()
                    index = hnswlib.Index(space='l2', dim=dim)
                    index.load_index(index_file, max_elements=len(chunks))
                    
                    self.subjects[subject_name] = {
                        "config": config,
                        "index": index,
                        "chunks": chunks
                    }
                    
            except Exception as e:
                continue
    
    def _load_config(self, config_file):
        """Загрузка конфига"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def detect_subject(self, question: str) -> List[str]:
        """Быстрая детекция предметов"""
        question_lower = question.lower()
        relevant = []
        
        # Проверяем основные ключевые слова
        math_keywords = ["производн", "интеграл", "предел", "дифференциал"]
        algebra_keywords = ["матриц", "вектор", "определитель", "линейн"]
        
        if any(kw in question_lower for kw in math_keywords):
            if "matan" in self.subjects:
                relevant.append("matan")
        
        if any(kw in question_lower for kw in algebra_keywords):
            if "linalg" in self.subjects:
                relevant.append("linalg")
        
        return relevant if relevant else list(self.subjects.keys())[:2]  # Ограничиваем
    
    def search_in_subject(self, subject_name: str, query: str, top_k: int = 3):
        """Оптимизированный поиск"""
        if subject_name not in self.subjects:
            return []
        
        subject_data = self.subjects[subject_name]
        query_emb = self.model.encode([query])[0]  # Берем первый элемент
        
        # Быстрый поиск
        indices, distances = subject_data["index"].knn_query(
            query_emb.reshape(1, -1), 
            k=min(top_k, len(subject_data["chunks"]))
        )
        
        return [subject_data["chunks"][idx] for idx in indices[0]]
    
    def ask(self, question: str, progress_callback=None) -> str:
        """Оптимизированный метод запроса"""
        if not self.subjects:
            return "❌ Нет загруженных учебных материалов."
        
        # Быстрый поиск контекста
        relevant_subjects = self.detect_subject(question)[:1]  # Берем только первый предмет
        
        all_contexts = []
        for subject_name in relevant_subjects:
            try:
                # Используем кэшированный поиск
                chunks = cached_search(self, subject_name, question, top_k=2)  # Только 2 чанка
                if chunks:
                    subject_title = self.subjects[subject_name]["config"]["subject"]
                    all_contexts.append(f"📘 {subject_title}:\n{chunks[0]}\n")
                    if len(chunks) > 1:
                        all_contexts.append(f"{chunks[1]}\n")
            except:
                continue
        
        # Формируем компактный промпт
        context = " ".join(all_contexts[:500])  # Ограничиваем контекст
        
        system_prompt = self._create_compact_prompt(context, question)
        
        # Быстрый запрос к API
        return self._make_fast_api_request(system_prompt, question)
    
    def _create_compact_prompt(self, context, question):
        """Компактный промпт"""
        if context:
            return f"""Ты — математик. Отвечай кратко и по делу.

Контекст: {context[:400]}...

Вопрос: {question}

Ответ (только суть, формулы в LaTeX):"""
        else:
            return f"""Ты — математик. Отвечай кратко.

Вопрос: {question}

Ответ (кратко, формулы в LaTeX \\(...\\)):"""
    
    def _make_fast_api_request(self, system_prompt, question, timeout=30):
        """Быстрый запрос к API с минимальными накладными расходами"""
        api_key = st.secrets.get("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY"))
        if not api_key:
            return "❌ API ключ не настроен."
        
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    "max_tokens": 800,  # Меньше токенов = быстрее
                    "temperature": 0.3,
                    "stream": False  # Без streaming для скорости
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"❌ Ошибка API: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "⏰ Таймаут. Попробуйте короче."
        except Exception as e:
            return f"❌ Ошибка: {str(e)}"

# ========== УПРОЩЕННЫЙ ИНТЕРФЕЙС ==========
def main():
    st.markdown('<h1 class="main-header">🎓 Математический Ассистент</h1>', unsafe_allow_html=True)
    
    # Инициализация
    if "assistant" not in st.session_state:
        with st.spinner("⚡ Загружаю материалы..."):
            st.session_state.assistant = get_assistant()
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    
    assistant = st.session_state.assistant
    
    # Боковая панель
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=80)
        
        st.markdown("### 📚 Предметы")
        if assistant.subjects:
            for subject_name, data in assistant.subjects.items():
                st.markdown(f"**{data['config']['subject']}**")
        else:
            st.warning("Нет предметов")
        
        st.markdown("---")
        st.markdown("### ⚡ Быстрые запросы")
        
        quick_queries = [
            "Что такое производная?",
            "Объясни интеграл",
            "Как найти предел?",
            "Что такое матрица?",
            "Правило Лопиталя",
            "Метод Гаусса"
        ]
        
        cols = st.columns(2)
        for idx, query in enumerate(quick_queries):
            with cols[idx % 2]:
                if st.button(query, key=f"quick_{idx}", 
                           use_container_width=True, 
                           type="secondary"):
                    st.session_state.question = query
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### 📜 История")
        
        if st.session_state.history:
            for i, item in enumerate(reversed(st.session_state.history[-3:])):
                st.caption(f"❓ {item['question'][:30]}...")
        else:
            st.caption("Нет истории")
    
    # Основная область
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "💭 Ваш вопрос:",
            value=st.session_state.get("question", ""),
            placeholder="Например: 'Что такое производная?'",
            key="question_input"
        )
    
    with col2:
        st.write("")  # Отступ
        st.write("")
        if st.button("🚀 Ответить", type="primary", use_container_width=True):
            if question.strip():
                # Показываем прогресс
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Быстрая обработка
                status_text.text("🔍 Ищу информацию...")
                progress_bar.progress(30)
                
                start_time = time.time()
                answer = assistant.ask(question)
                elapsed = time.time() - start_time
                
                progress_bar.progress(70)
                status_text.text("📝 Форматирую ответ...")
                
                # Сохраняем
                st.session_state.history.append({
                    "question": question,
                    "answer": answer,
                    "time": elapsed
                })
                
                st.session_state.last_answer = answer
                st.session_state.last_time = elapsed
                
                progress_bar.progress(100)
                status_text.text(f"✅ Готово за {elapsed:.1f} сек")
                
                st.rerun()
            else:
                st.warning("Введите вопрос")
    
    # Отображение ответа
    if st.session_state.last_answer:
        st.markdown("---")
        
        if "last_time" in st.session_state:
            st.caption(f"⏱️ Ответ получен за {st.session_state.last_time:.1f} сек")
        
        # Отображение с LaTeX
        st.markdown(f"""
        <div class="math-content">
            {st.session_state.last_answer}
        </div>
        """, unsafe_allow_html=True)
        
        # Быстрые действия
        col_a, col_b, col_c = st.columns([1, 1, 1])
        
        with col_a:
            if st.button("🔄 Новый вопрос", use_container_width=True):
                st.session_state.question = ""
                st.session_state.last_answer = None
                st.rerun()
        
        with col_b:
            if st.button("📋 Копировать", use_container_width=True):
                st.code(st.session_state.last_answer)
        
        with col_c:
            if st.button("💾 Сохранить", use_container_width=True):
                # Сохранение в файл
                filename = f"ответ_{int(time.time())}.md"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"# Вопрос:\n{st.session_state.get('question', '')}\n\n")
                    f.write(f"# Ответ:\n{st.session_state.last_answer}")
                st.success(f"Сохранено в {filename}")
    
    # Быстрый пример
    if not st.session_state.last_answer and not st.session_state.get("question"):
        st.markdown("---")
        st.info("💡 **Совет:** Задайте конкретный вопрос для быстрого ответа")
        
        example_cols = st.columns(3)
        examples = [
            ("Производная функции f(x)=x²", "f'(x) = 2x"),
            ("Интеграл от x dx", "∫x dx = x²/2 + C"),
            ("Определитель матрицы 2x2", "det([[a,b],[c,d]]) = ad - bc")
        ]
        
        for idx, (ex_q, ex_a) in enumerate(examples):
            with example_cols[idx]:
                if st.button(f"Пример {idx+1}", key=f"ex_{idx}"):
                    st.session_state.question = ex_q
                    st.session_state.last_answer = f"**Ответ:** \\({ex_a}\\)"
                    st.rerun()

# Запуск приложения
if __name__ == "__main__":
    main()