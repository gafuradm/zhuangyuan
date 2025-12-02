import streamlit as st
import os
import json
import requests
import numpy as np
import hnswlib
from typing import List
import time
import hashlib
import re
import html

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
<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 1rem; }
    .subject-card { background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #3B82F6; }
    .math-answer { 
        font-size: 1.1em; 
        line-height: 1.6; 
        margin: 1em 0;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
    }
    .katex { font-size: 1.1em !important; }
    .katex-display { margin: 1em 0 !important; padding: 1em; background-color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ========== УТИЛИТЫ ДЛЯ РЕНДЕРИНГА ==========
def clean_latex_content(text: str) -> str:
    """Очищает текст от лишних символов и форматирует LaTeX"""
    # Удаляем лишние пробелы и переносы
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Заменяем [ и ] на $$ для блочных формул
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    
    # Заменяем \( и \) на $ для строчных формул
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    
    # Экранируем HTML-сущности, но оставляем LaTeX
    text = html.escape(text)
    
    # Восстанавливаем LaTeX команды
    latex_patterns = [
        (r'&amp;lt;', '<'),
        (r'&amp;gt;', '>'),
        (r'&amp;quot;', '"'),
        (r'&amp;amp;', '&'),
        (r'&lt;', '<'),
        (r'&gt;', '>'),
        (r'&quot;', '"'),
        (r'&amp;', '&'),
    ]
    
    for pattern, replacement in latex_patterns:
        text = text.replace(pattern, replacement)
    
    return text

def render_with_katex(text: str) -> str:
    """Оборачивает текст для рендеринга KaTeX"""
    cleaned_text = clean_latex_content(text)
    
    # Добавляем скрипт для рендеринга KaTeX
    html_content = f"""
    <div class="math-answer" id="math-content-{hash(text)}">
        {cleaned_text}
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const element = document.getElementById('math-content-{hash(text)}');
            if (element && window.renderMathInElement) {{
                renderMathInElement(element, {{
                    delimiters: [
                        {{left: '$$', right: '$$', display: true}},
                        {{left: '$', right: '$', display: false}},
                        {{left: '\\\\(', right: '\\\\)', display: false}},
                        {{left: '\\\\[', right: '\\\\]', display: true}}
                    ],
                    throwOnError: false,
                    trust: true
                }});
            }}
        }});
        
        // Также рендерим при изменении контента
        setTimeout(function() {{
            const element = document.getElementById('math-content-{hash(text)}');
            if (element && window.renderMathInElement) {{
                renderMathInElement(element, {{
                    delimiters: [
                        {{left: '$$', right: '$$', display: true}},
                        {{left: '$', right: '$', display: false}},
                        {{left: '\\\\(', right: '\\\\)', display: false}},
                        {{left: '\\\\[', right: '\\\\]', display: true}}
                    ],
                    throwOnError: false,
                    trust: true
                }});
            }}
        }}, 100);
    </script>
    """
    
    return html_content

# ========== МОДЕЛЬ ЭМБЕДДИНГОВ ==========
class SimpleEmbedder:
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

# ========== ОСНОВНОЙ КЛАСС ==========
class MathAssistant:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.model = SimpleEmbedder(dim=384)
        self.subjects = {}
        self.load_subjects()
    
    def load_subjects(self):
        if not os.path.exists(self.data_dir):
            return
        
        subject_folders = [d for d in os.listdir(self.data_dir) 
                          if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for subject_name in subject_folders:
            try:
                subject_path = os.path.join(self.data_dir, subject_name)
                required_files = ["config.json", "index.hnsw", "chunks.npy"]
                
                if not all(os.path.exists(os.path.join(subject_path, f)) for f in required_files):
                    continue
                
                with open(os.path.join(subject_path, "config.json"), 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                chunks = np.load(os.path.join(subject_path, "chunks.npy"), allow_pickle=True)
                dim = self.model.get_sentence_embedding_dimension()
                index = hnswlib.Index(space='l2', dim=dim)
                index.load_index(os.path.join(subject_path, "index.hnsw"), max_elements=len(chunks))
                
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
        subject_data = self.subjects[subject_name]
        query_emb = self.model.encode([query])
        indices, distances = subject_data["index"].knn_query(query_emb, k=top_k)
        return [subject_data["chunks"][idx] for idx in indices[0]]
    
    def ask(self, question: str) -> str:
        if not self.subjects:
            return "❌ Нет загруженных учебных материалов."
        
        relevant_subjects = self.detect_subject(question)
        all_contexts = []
        
        for subject_name in relevant_subjects:
            try:
                chunks = self.search_in_subject(subject_name, question, top_k=3)
                subject_title = self.subjects[subject_name]["config"]["subject"]
                for chunk in chunks[:3]:
                    all_contexts.append(f"📘 {subject_title}:\n{chunk}\n")
            except Exception:
                continue
        
        context = "\n".join(all_contexts)
        
        if context.strip():
            system_prompt = f"""Ты — преподаватель математики. Отвечай на русском языке.

ИСПОЛЬЗУЙ ТОЛЬКО ЭТИ ФОРМАТЫ ДЛЯ ФОРМУЛ:
- Для встроенных формул: $формула$
- Для вынесенных формул: $$формула$$

НЕ ИСПОЛЬЗУЙ: \\(, \\), \\[, \\]

Пример правильного ответа:
Производная функции f(x) = x^2 равна $f'(x) = 2x$.
Интеграл от функции вычисляется так:
$$\\int x^2 dx = \\frac{x^3}{3} + C$$

ИНФОРМАЦИЯ ИЗ УЧЕБНИКОВ:
{context}

ВОПРОС: {question}

ОТВЕТ (только на русском, формулы в формате $...$ или $$...$$):
"""
        else:
            system_prompt = f"""Ты — преподаватель математики. Отвечай понятно и подробно на русском языке.

ИСПОЛЬЗУЙ ТОЛЬКО ЭТИ ФОРМАТЫ ДЛЯ ФОРМУЛ:
- Для встроенных формул: $формула$
- Для вынесенных формул: $$формула$$

ВОПРОС: {question}

ОТВЕТ:
"""
        
        api_key = st.secrets.get("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY"))
        if not api_key:
            return "❌ API ключ не настроен."
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "max_tokens": 2000,
            "temperature": 0.3
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
                return f"❌ Ошибка API ({response.status_code})"
                
        except Exception as e:
            return f"❌ Ошибка соединения: {str(e)}"

# ========== ИНТЕРФЕЙС STREAMLIT ==========
def main():
    st.markdown('<h1 class="main-header">🎓 Математический Ассистент</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-помощник по математике на основе ваших учебников</p>', unsafe_allow_html=True)
    
    if "assistant" not in st.session_state:
        with st.spinner("🔄 Загружаю учебные материалы..."):
            st.session_state.assistant = MathAssistant("data")
    
    assistant = st.session_state.assistant
    
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=100)
        st.markdown("### 📚 Загруженные предметы")
        
        if assistant.subjects:
            for subject_name, data in assistant.subjects.items():
                st.markdown(f"""
                <div class="subject-card">
                <strong>{data['config']['subject']}</strong><br>
                📖 {len(data['config']['books'])} книг<br>
                🧩 {len(data['chunks'])} фрагментов
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Учебные материалы не загружены")
        
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
                if "last_answer" in st.session_state:
                    del st.session_state.last_answer
                st.rerun()
    
    st.markdown("### 💭 Задайте вопрос по математике")
    
    question = st.text_area(
        "Введите ваш вопрос:",
        value=st.session_state.get("question", ""),
        placeholder="Например: 'Что такое производная?' или 'Объясни метод Гаусса'",
        height=100,
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Получить ответ", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("🔍 Ищу информацию в учебниках..."):
                    start_time = time.time()
                    answer = assistant.ask(question)
                    elapsed = time.time() - start_time
                    
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    
                    st.session_state.history.append({
                        "question": question,
                        "answer": answer,
                        "time": elapsed
                    })
                    
                    st.session_state.last_answer = answer
                    st.session_state.last_time = elapsed
                    st.rerun()
            else:
                st.warning("⚠️ Введите вопрос")
    
    with col2:
        if st.button("🔄 Новый вопрос", use_container_width=True):
            if "last_answer" in st.session_state:
                del st.session_state.last_answer
            st.session_state.question = ""
            st.rerun()
    
    if "last_answer" in st.session_state:
        st.markdown(f"### 📚 Ответ ({st.session_state.get('last_time', 0):.1f} сек)")
        st.markdown("---")
        
        # Отображаем ответ с KaTeX
        st.markdown(render_with_katex(st.session_state.last_answer), unsafe_allow_html=True)
        
        # Отладочная информация (опционально)
        with st.expander("📄 Исходный текст ответа"):
            st.text(st.session_state.last_answer)
    
    # История
    if st.sidebar.button("📜 Показать историю", use_container_width=True):
        if "history" in st.session_state and st.session_state.history:
            st.sidebar.markdown("### 📜 История вопросов")
            for i, item in enumerate(reversed(st.session_state.history[-5:])):
                with st.sidebar.expander(f"❓ {item['question'][:50]}..."):
                    st.write(f"**Время:** {item['time']:.1f} сек")
                    st.markdown(render_with_katex(item["answer"][:300] + ("..." if len(item["answer"]) > 300 else "")), unsafe_allow_html=True)
        else:
            st.sidebar.info("📝 История вопросов пуста")
    
    # Информация
    with st.sidebar.expander("ℹ️ О системе"):
        st.markdown("""
        **Формулы должны быть в формате:**
        - Встроенные: `$формула$`
        - Вынесенные: `$$формула$$`
        
        **Примеры:**
        - $E = mc^2$
        - $$\\int_a^b f(x) dx$$
        """)
        
        if st.button("🧪 Тест KaTeX", key="test_katex"):
            test_answer = """
            **Тест формул:**
            
            Встроенная формула: $E = mc^2$
            
            Вынесенная формула:
            $$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$
            
            Производная: $f'(x) = \\lim_{h \\to 0} \\frac{f(x+h)-f(x)}{h}$
            
            Матрица: $\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$
            """
            st.session_state.last_answer = test_answer
            st.rerun()

if __name__ == "__main__":
    main()