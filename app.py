import streamlit as st
import os
import json
import requests
import numpy as np
import hnswlib
from typing import List
import time
import hashlib
import uuid
import threading
from queue import Queue
from datetime import datetime
import signal

# ========== КОНФИГУРАЦИЯ ==========
st.set_page_config(
    page_title="Математический Ассистент",
    page_icon="📚",
    layout="wide"
)

# Загружаем KaTeX в самом начале
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" integrity="sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous"></script>
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
    .stButton button {
        width: 100%;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    /* Стили для математического контента */
    .math-content {
        font-size: 1.1em;
        line-height: 1.8;
        margin: 1em 0;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
    }
    .math-content p {
        margin-bottom: 1em;
    }
    .katex-display {
        margin: 1.5em 0 !important;
        padding: 1em;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        overflow-x: auto;
        overflow-y: hidden;
    }
    .katex {
        font-size: 1.1em !important;
        padding: 2px 4px;
    }
    .progress-container {
        margin: 20px 0;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
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
            text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(text_hash)
            emb = np.random.randn(self.dim).astype(np.float32)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def get_sentence_embedding_dimension(self):
        return self.dim

# ========== МЕНЕДЖЕР ИСТОРИИ ==========
class HistoryManager:
    """Менеджер истории с сохранением в файл"""
    
    def __init__(self, filename="history.json", max_entries=100):
        self.filename = filename
        self.max_entries = max_entries
        self.history = self.load_history()
    
    def load_history(self):
        """Загрузка истории из файла"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Преобразуем строки дат обратно в объекты datetime
                    for entry in data:
                        entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                    return data
        except Exception as e:
            print(f"Ошибка загрузки истории: {e}")
        return []
    
    def save_history(self):
        """Сохранение истории в файл"""
        try:
            # Преобразуем datetime в строки для JSON
            data_to_save = []
            for entry in self.history[:self.max_entries]:
                entry_copy = entry.copy()
                entry_copy['timestamp'] = entry['timestamp'].isoformat()
                data_to_save.append(entry_copy)
            
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения истории: {e}")
    
    def add_entry(self, question, answer, elapsed_time):
        """Добавление записи в историю"""
        entry = {
            "id": str(uuid.uuid4()),
            "question": question,
            "answer": answer,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now()
        }
        
        self.history.insert(0, entry)  # Добавляем в начало
        
        # Ограничиваем количество записей
        if len(self.history) > self.max_entries:
            self.history = self.history[:self.max_entries]
        
        # Сохраняем в файл
        self.save_history()
        return entry
    
    def get_recent(self, count=10):
        """Получение последних записей"""
        return self.history[:count]
    
    def clear_history(self):
        """Очистка истории"""
        self.history = []
        self.save_history()
    
    def get_by_id(self, entry_id):
        """Получение записи по ID"""
        for entry in self.history:
            if entry['id'] == entry_id:
                return entry
        return None

# ========== АСИНХРОННЫЙ ПРОЦЕССОР ==========
class AsyncProcessor:
    """Асинхронный процессор для долгих запросов"""
    def __init__(self):
        self.task_queue = Queue()
        self.results = {}
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def _worker(self):
        """Рабочий поток для обработки задач"""
        while self.is_running:
            try:
                task_id, question, assistant, callback = self.task_queue.get(timeout=1)
                try:
                    # Обработка с прогрессом
                    result = self._process_with_progress(question, assistant, callback)
                    self.results[task_id] = {
                        "status": "completed",
                        "result": result,
                        "timestamp": datetime.now()
                    }
                except TimeoutError:
                    self.results[task_id] = {
                        "status": "timeout",
                        "result": "⏰ Время обработки истекло. Попробуйте задать более конкретный вопрос.",
                        "timestamp": datetime.now()
                    }
                except Exception as e:
                    self.results[task_id] = {
                        "status": "error",
                        "result": f"❌ Ошибка: {str(e)}",
                        "timestamp": datetime.now()
                    }
                finally:
                    if callback:
                        callback(1.0, "✅ Завершено")
                self.task_queue.task_done()
            except Queue.Empty:
                continue
    
    def _process_with_progress(self, question, assistant, progress_callback):
        """Обработка с прогрессом и таймаутом"""
        class TimeoutException(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutException()
        
        # Устанавливаем обработчик таймаута
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(180)  # 3 минуты таймаут
        
        try:
            return assistant.ask_with_progress(question, progress_callback)
        except TimeoutException:
            raise TimeoutError("Время обработки истекло")
        finally:
            signal.alarm(0)  # Отключаем таймер
    
    def submit_task(self, question, assistant, progress_callback=None):
        """Добавление задачи в очередь"""
        task_id = str(uuid.uuid4())
        self.task_queue.put((task_id, question, assistant, progress_callback))
        return task_id
    
    def get_result(self, task_id):
        """Получение результата"""
        return self.results.get(task_id)
    
    def cleanup_old_results(self, max_age_minutes=30):
        """Очистка старых результатов"""
        now = datetime.now()
        to_delete = []
        for task_id, result in self.results.items():
            if (now - result["timestamp"]).seconds > max_age_minutes * 60:
                to_delete.append(task_id)
        for task_id in to_delete:
            del self.results[task_id]

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
                
                required_files = ["config.json", "index.hnsw", "chunks.npy"]
                if not all(os.path.exists(os.path.join(subject_path, f)) for f in required_files):
                    st.warning(f"⚠️ В папке '{subject_name}' не хватает файлов")
                    continue
                
                with open(os.path.join(subject_path, "config.json"), 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                chunks = np.load(os.path.join(subject_path, "chunks.npy"), allow_pickle=True)
                
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
    
    def ask_with_progress(self, question: str, progress_callback=None) -> str:
        """Улучшенный метод с поддержкой прогресса"""
        if not self.subjects:
            return "❌ Нет загруженных учебных материалов."
        
        # Шаг 1: Детекция предметов
        if progress_callback:
            progress_callback(0.05, "🔍 Анализирую вопрос...")
        
        relevant_subjects = self.detect_subject(question)
        
        # Шаг 2: Поиск по предметам
        all_contexts = []
        total_subjects = len(relevant_subjects)
        
        for idx, subject_name in enumerate(relevant_subjects):
            progress = 0.05 + (0.25 * (idx / max(total_subjects, 1)))
            if progress_callback:
                progress_callback(progress, f"📚 Ищу в {subject_name}...")
            
            try:
                chunks = self.search_in_subject(subject_name, question, top_k=3)
                subject_title = self.subjects[subject_name]["config"]["subject"]
                for i, chunk in enumerate(chunks[:3]):
                    all_contexts.append(f"📘 {subject_title}:\n{chunk}\n")
                
                # Небольшая задержка для обновления UI
                time.sleep(0.05)
                
            except Exception as e:
                continue
        
        # Шаг 3: Формирование промпта
        if progress_callback:
            progress_callback(0.3, "📝 Формирую запрос...")
        
        context = "\n".join(all_contexts)
        
        if context.strip():
            system_prompt = self._create_prompt_with_context(context, question)
        else:
            system_prompt = self._create_general_prompt(question)
        
        # Шаг 4: Запрос к API
        if progress_callback:
            progress_callback(0.35, "🤖 Обращаюсь к нейросети...")
        
        # Запрос с повторными попытками
        response = self._make_api_request_with_retry(
            system_prompt=system_prompt,
            question=question,
            max_retries=3,
            progress_callback=progress_callback
        )
        
        return response
    
    def _make_api_request_with_retry(self, system_prompt, question, max_retries=3, progress_callback=None):
        """Запрос к API с повторными попытками"""
        api_key = st.secrets.get("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY"))
        if not api_key:
            return "❌ API ключ не настроен. Добавьте DEEPSEEK_API_KEY в секреты Streamlit."
        
        for attempt in range(max_retries):
            try:
                if progress_callback:
                    progress = 0.35 + (0.6 * (attempt / max_retries))
                    progress_callback(progress, f"🔄 Попытка {attempt + 1}/{max_retries}...")
                
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.3
                }
                
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=60  # Таймаут 60 секунд
                )
                
                if response.status_code == 200:
                    if progress_callback:
                        progress_callback(0.95, "✅ Получен ответ...")
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    if attempt == max_retries - 1:
                        return f"❌ Ошибка API ({response.status_code}): {response.text}"
                    time.sleep(2 ** attempt)  # Экспоненциальная backoff
                    
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    return "⏰ Таймаут при подключении к API"
                time.sleep(2)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"❌ Ошибка соединения: {str(e)}"
                time.sleep(1)
        
        return "❌ Не удалось получить ответ после нескольких попыток"
    
    def _create_prompt_with_context(self, context, question):
        """Создание промпта с контекстом"""
        return f"""Ты — преподаватель математики. Отвечай на русском языке.

ВАЖНО: Все математические формулы должны быть записаны в формате LaTeX:
- Для формул в строке: \\(формула\\)
- Для вынесенных формул: $$формула$$
- Используй стандартные обозначения LaTeX

Пример:
Производная функции: \\(f'(x) = \\lim_{{h \\to 0}} \\frac{{f(x+h)-f(x)}}{{h}}\\)
Интеграл: $$\\int_a^b f(x) dx$$

ИНФОРМАЦИЯ ИЗ УЧЕБНИКОВ:
{context}

ВОПРОС: {question}

ОТВЕТ (обязательно используй LaTeX для всех математических выражений, отвечай подробно и понятно):
"""
    
    def _create_general_prompt(self, question):
        """Создание общего промпта"""
        return f"""Ты — преподаватель математики. Отвечай понятно и подробно на русском языке.

ВСЕ математические формулы записывай в LaTeX:
- Встроенные: \\(формула\\)
- Вынесенные: $$формула$$

ВОПРОС: {question}

ОТВЕТ (отвечай максимально подробно с примерами, используй LaTeX для формул):
"""

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
def render_math_answer(answer: str):
    """Отображает ответ с поддержкой LaTeX"""
    html = f"""
    <div class="math-content">
        {answer}
    </div>
    <script>
        // Перерендерим формулы после загрузки контента
        if (window.renderMathInElement) {{
            renderMathInElement(document.querySelector('.math-content'), {{
                delimiters: [
                    {{left: '$$', right: '$$', display: true}},
                    {{left: '$', right: '$', display: false}},
                    {{left: '\\\\(', right: '\\\\)', display: false}},
                    {{left: '\\\\[', right: '\\\\]', display: true}}
                ],
                throwOnError: false
            }});
        }}
    </script>
    """
    return html

def update_progress(progress_value, progress_text):
    """Обновление прогресса в session_state"""
    st.session_state.last_progress = (progress_value, progress_text)

# ========== ИНТЕРФЕЙС STREAMLIT ==========
def main():
    st.markdown('<h1 class="main-header">🎓 Математический Ассистент</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-помощник по математике на основе ваших учебников</p>', unsafe_allow_html=True)
    
    # Инициализация компонентов
    if "assistant" not in st.session_state:
        with st.spinner("🔄 Загружаю учебные материалы..."):
            st.session_state.assistant = MathAssistant("data")
    
    if "history_manager" not in st.session_state:
        st.session_state.history_manager = HistoryManager()
    
    if "async_processor" not in st.session_state:
        st.session_state.async_processor = AsyncProcessor()
    
    if "processing_task_id" not in st.session_state:
        st.session_state.processing_task_id = None
    
    if "last_progress" not in st.session_state:
        st.session_state.last_progress = (0, "")
    
    assistant = st.session_state.assistant
    history_manager = st.session_state.history_manager
    
    # Проверяем выполнение асинхронных задач
    if st.session_state.processing_task_id:
        result = st.session_state.async_processor.get_result(
            st.session_state.processing_task_id
        )
        
        if result:
            if result["status"] in ["completed", "timeout", "error"]:
                # Сохраняем в историю
                if "last_question" in st.session_state:
                    history_manager.add_entry(
                        question=st.session_state.last_question,
                        answer=result["result"],
                        elapsed_time=st.session_state.get("last_elapsed", 0)
                    )
                
                st.session_state.last_answer = result["result"]
                st.session_state.processing_task_id = None
                st.session_state.last_progress = (0, "")
                st.rerun()
    
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
            st.info("""
            Создайте структуру:
            ```
            data/
            ├── matan/
            │   ├── config.json
            │   ├── index.hnsw
            │   └── chunks.npy
            └── linalg/
                ├── config.json
                ├── index.hnsw
                └── chunks.npy
            ```
            """)
        
        st.markdown("---")
        st.markdown("### 📜 История вопросов")
        
        # Показываем историю из файла
        recent_history = history_manager.get_recent(5)
        if recent_history:
            for entry in recent_history:
                with st.expander(f"❓ {entry['question'][:50]}...", expanded=False):
                    st.markdown(f"**Время:** {entry['timestamp'].strftime('%H:%M')}")
                    st.markdown(f"**Длительность:** {entry['elapsed_time']:.1f} сек")
                    if st.button("↩️ Повторить", key=f"repeat_{entry['id']}"):
                        st.session_state.question = entry['question']
                        st.rerun()
        else:
            st.info("📝 История вопросов пуста")
        
        st.markdown("---")
        st.markdown("### 💡 Примеры вопросов")
        
        examples = [
            "Что такое производная?",
            "Как найти определитель матрицы?",
            "Объясни правило Лопиталя",
            "Что такое собственные значения?"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{hash(example)}"):
                st.session_state.question = example
                st.rerun()
        
        # Кнопка очистки истории
        if st.button("🗑️ Очистить историю", type="secondary"):
            history_manager.clear_history()
            st.success("История очищена!")
            time.sleep(1)
            st.rerun()
    
    # Основная область
    st.markdown("### 💭 Задайте вопрос по математике")
    
    question = st.text_area(
        "Введите ваш вопрос:",
        value=st.session_state.get("question", ""),
        placeholder="Например: 'Что такое производная?' или 'Объясни метод Гаусса'",
        height=120,
        label_visibility="collapsed",
        key="question_input"
    )
    
    # Прогресс-бар для долгих запросов
    if st.session_state.processing_task_id:
        progress_value, progress_text = st.session_state.last_progress
        if progress_value > 0:
            st.markdown(f"""
            <div class="progress-container">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>⏳ Обработка...</span>
                    <span>{int(progress_value * 100)}%</span>
                </div>
                <div style="height: 10px; background: rgba(255,255,255,0.3); border-radius: 5px; overflow: hidden;">
                    <div style="width: {progress_value * 100}%; height: 100%; background: white; transition: width 0.3s;"></div>
                </div>
                <div style="margin-top: 10px; font-size: 0.9em;">{progress_text}</div>
            </div>
            """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        is_processing = st.session_state.processing_task_id is not None
        if st.button("🎯 Получить ответ", type="primary", use_container_width=True,
                    disabled=is_processing):
            if question.strip():
                # Сохраняем вопрос
                st.session_state.last_question = question
                st.session_state.question_start_time = time.time()
                
                # Запускаем асинхронную обработку
                task_id = st.session_state.async_processor.submit_task(
                    question, assistant, update_progress
                )
                st.session_state.processing_task_id = task_id
                st.session_state.last_progress = (0.05, "Начинаю обработку...")
                
                st.rerun()
            else:
                st.warning("⚠️ Введите вопрос")
    
    with col2:
        if st.button("🔄 Новый вопрос", use_container_width=True,
                    disabled=is_processing):
            # Сбрасываем состояние
            keys_to_reset = ["last_answer", "processing_task_id", 
                           "last_progress", "question"]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    with col3:
        if st.button("⏹️ Остановить", use_container_width=True,
                    disabled=not is_processing):
            if st.session_state.processing_task_id:
                # Помечаем задачу как отмененную
                st.session_state.async_processor.results[st.session_state.processing_task_id] = {
                    "status": "cancelled",
                    "result": "❌ Запрос отменен пользователем",
                    "timestamp": datetime.now()
                }
                st.session_state.processing_task_id = None
                st.session_state.last_progress = (0, "")
                st.success("Запрос остановлен")
                time.sleep(1)
                st.rerun()
    
    # Автоматическое обновление прогресса
    if st.session_state.processing_task_id and st.session_state.last_progress[0] < 0.95:
        # Имитация прогресса для UI
        current_progress, current_text = st.session_state.last_progress
        if current_progress < 0.8:
            new_progress = min(0.8, current_progress + 0.02)
            progress_stages = [
                (0.1, "🔍 Анализирую вопрос..."),
                (0.25, "📚 Ищу в учебниках..."),
                (0.4, "🤖 Формирую запрос..."),
                (0.6, "🌐 Отправляю запрос..."),
                (0.8, "📝 Получаю ответ...")
            ]
            
            # Выбираем текст на основе прогресса
            new_text = current_text
            for stage_progress, stage_text in progress_stages:
                if new_progress >= stage_progress:
                    new_text = stage_text
            
            st.session_state.last_progress = (new_progress, new_text)
        
        # Автоматическое обновление каждые 1.5 секунды
        time.sleep(1.5)
        st.rerun()
    
    # Отображение ответа
    if "last_answer" in st.session_state and not st.session_state.processing_task_id:
        elapsed = st.session_state.get("last_elapsed", 0)
        if "question_start_time" in st.session_state:
            elapsed = time.time() - st.session_state.question_start_time
            st.session_state.last_elapsed = elapsed
        
        st.markdown(f"### 📚 Ответ ({elapsed:.1f} сек)")
        st.markdown("---")
        
        # Отображаем ответ с поддержкой LaTeX
        st.markdown(render_math_answer(st.session_state.last_answer), unsafe_allow_html=True)
        
        # Кнопка для копирования
        if st.button("📋 Скопировать ответ"):
            st.code(st.session_state.last_answer, language="markdown")
            st.success("Ответ скопирован в буфер обмена (в виде текста)")
    
    # Информация о системе
    with st.expander("ℹ️ О системе"):
        st.markdown("""
        **Как работает система:**
        1. 📚 Загружает ваши учебники (PDF → текст)
        2. 🔍 Ищет релевантные фрагменты по вопросу
        3. 🤖 Отправляет контекст в DeepSeek AI
        4. 📝 Получает подробный ответ
        
        **Особенности:**
        - ⏳ Поддержка долгих запросов (до 3 минут)
        - 💾 История сохраняется между перезагрузками
        - 📊 Прогресс-бар для мониторинга
        - ✋ Возможность остановки запроса
        
        **Требования:**
        - DeepSeek API ключ (добавьте в секреты Streamlit)
        - Папка `data/` с индексами учебников
        
        **LaTeX поддержка:**
        - Все формулы автоматически рендерятся с помощью KaTeX
        - Используйте \\(формула\\) для встроенных формул
        - Используйте $$формула$$ для вынесенных формул
        """)
        
        if st.button("🧪 Проверить LaTeX рендеринг"):
            test_math = r"""
            **Тест математических формул:**
            
            Встроенная формула: \(E = mc^2\)
            
            Формула на отдельной строке:
            $$
            \int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
            $$
            
            Производная: $$\frac{dy}{dx} = \lim_{\Delta x \to 0} \frac{f(x+\Delta x) - f(x)}{\Delta x}$$
            
            Матрица: $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$
            
            Сумма: \(\sum_{i=1}^{n} i = \frac{n(n+1)}{2}\)
            """
            st.markdown(render_math_answer(test_math), unsafe_allow_html=True)

if __name__ == "__main__":
    main()