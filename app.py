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
    page_title="Math Assistant",
    page_icon="📚",
    layout="wide"
)

# ========== ТЕМНАЯ ТЕМА ==========
def apply_custom_theme(dark_mode=True):
    """Применяет кастомную тему"""
    if dark_mode:
        theme_css = """
        <style>
        /* Основные стили для темной темы */
        .stApp {
            background-color: #0E1117;
        }
        
        /* Заголовки */
        h1, h2, h3, h4, h5, h6 {
            color: #60A5FA !important;
        }
        
        .main-header {
            font-size: 2.5rem;
            color: #60A5FA !important;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        /* Основной текст */
        .stMarkdown, .stText, p, div {
            color: #FAFAFA !important;
        }
        
        /* Карточки предметов */
        .subject-card {
            background: #1E293B;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #3B82F6;
            color: #E2E8F0 !important;
        }
        
        .subject-card strong {
            color: #60A5FA !important;
        }
        
        /* Математический контент */
        .math-content {
            font-size: 1.1em;
            line-height: 1.8;
            margin: 1em 0;
            padding: 20px;
            background-color: #1E293B;
            border-radius: 10px;
            border-left: 4px solid #3B82F6;
            color: #E2E8F0 !important;
        }
        
        .math-content p {
            margin-bottom: 1em;
            color: #E2E8F0 !important;
        }
        
        /* KaTeX */
        .katex-display {
            margin: 1.5em 0 !important;
            padding: 1em;
            background-color: #0F172A;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            overflow-x: auto;
            overflow-y: hidden;
        }
        
        /* Текстовое поле */
        .stTextArea textarea {
            background-color: #1E293B !important;
            color: #FAFAFA !important;
            border: 1px solid #334155 !important;
        }
        
        /* Кнопки */
        .stButton button {
            width: 100%;
            transition: all 0.3s;
            background-color: #1E40AF !important;
            color: white !important;
            border: none !important;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(59, 130, 246, 0.3) !important;
            background-color: #2563EB !important;
        }
        
        .stButton button[kind="primary"] {
            background-color: #2563EB !important;
        }
        
        .stButton button[kind="primary"]:hover {
            background-color: #3B82F6 !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #1E293B !important;
            color: #E2E8F0 !important;
            border: 1px solid #334155 !important;
        }
        
        .streamlit-expanderContent {
            background-color: #1E293B !important;
            color: #E2E8F0 !important;
        }
        
        /* Сайдбар */
        [data-testid="stSidebar"] {
            background-color: #0F172A !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: #E2E8F0 !important;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #60A5FA !important;
        }
        
        /* Предупреждения и информация */
        .stAlert {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
        }
        
        /* Иконки */
        .st-emotion-cache-1v0mbdj {
            filter: brightness(0.8);
        }
        
        /* Секции */
        .stMarkdown h3 {
            color: #60A5FA !important;
            border-bottom: 2px solid #3B82F6;
            padding-bottom: 0.5rem;
        }
        
        /* Разделитель */
        hr {
            border-color: #334155 !important;
        }
        
        </style>
        """
    else:
        theme_css = """
        <style>
        /* Светлая тема */
        .stApp {
            background-color: #FFFFFF;
        }
        
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A !important;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .stMarkdown, .stText, p, div {
            color: #0F172A !important;
        }
        
        .subject-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #3B82F6;
            color: #0F172A !important;
        }
        
        .subject-card strong {
            color: #1E40AF !important;
        }
        
        .math-content {
            font-size: 1.1em;
            line-height: 1.8;
            margin: 1em 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #3B82F6;
            color: #0F172A !important;
        }
        
        .katex-display {
            margin: 1.5em 0 !important;
            padding: 1em;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stTextArea textarea {
            background-color: white !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
        }
        
        .streamlit-expanderHeader {
            background-color: #f8f9fa !important;
            color: #0F172A !important;
            border: 1px solid #E2E8F0 !important;
        }
        
        .streamlit-expanderContent {
            background-color: #f8f9fa !important;
            color: #0F172A !important;
        }
        
        [data-testid="stSidebar"] {
            background-color: #f8fafc !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: #0F172A !important;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #1E40AF !important;
        }
        
        .stAlert {
            background-color: #f8f9fa !important;
            border: 1px solid #E2E8F0 !important;
        }
        
        .stMarkdown h3 {
            color: #1E40AF !important;
            border-bottom: 2px solid #3B82F6;
            padding-bottom: 0.5rem;
        }
        
        hr {
            border-color: #E2E8F0 !important;
        }
        
        </style>
        """
    
    st.markdown(theme_css, unsafe_allow_html=True)

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

# ========== МОДЕЛЬ ЭМБЕДДИНГОВ ==========
class SimpleEmbedder:
    """Simple offline model"""
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
        """Loads all subjects"""
        if not os.path.exists(self.data_dir):
            st.error(f"❌ Folder '{self.data_dir}' not found!")
            return
        
        subject_folders = [d for d in os.listdir(self.data_dir) 
                          if os.path.isdir(os.path.join(self.data_dir, d))]
        
        if not subject_folders:
            st.warning("⚠️ No subjects found in data/ folder")
            return
        
        for subject_name in subject_folders:
            try:
                subject_path = os.path.join(self.data_dir, subject_name)
                
                required_files = ["config.json", "index.hnsw", "chunks.npy"]
                if not all(os.path.exists(os.path.join(subject_path, f)) for f in required_files):
                    st.warning(f"⚠️ Missing files in '{subject_name}' folder")
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
                st.error(f"❌ Error loading '{subject_name}': {str(e)}")
    
    def detect_subject(self, question: str) -> List[str]:
        """Detects the subject of the question"""
        question_lower = question.lower()
        subject_keywords = {
            "matan": ["calculus", "derivative", "integral", 
                     "limit", "series", "function", "differentiation",
                     "differential", "math analysis"],
            "linalg": ["linear", "matrix", "vector", "determinant", 
                      "eigen", "linear space", "algebra", "linear algebra"]
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
        """Searches within a specific subject"""
        subject_data = self.subjects[subject_name]
        query_emb = self.model.encode([query])
        indices, distances = subject_data["index"].knn_query(query_emb, k=top_k)
        return [subject_data["chunks"][idx] for idx in indices[0]]
    
    def ask(self, question: str) -> str:
        """Main method for answering questions"""
        if not self.subjects:
            return "❌ No learning materials loaded."
        
        relevant_subjects = self.detect_subject(question)
        
        all_contexts = []
        for subject_name in relevant_subjects:
            try:
                chunks = self.search_in_subject(subject_name, question, top_k=3)
                subject_title = self.subjects[subject_name]["config"]["subject"]
                for i, chunk in enumerate(chunks[:3]):
                    all_contexts.append(f"📘 {subject_title}:\n{chunk}\n")
            except Exception as e:
                continue
        
        context = "\n".join(all_contexts)
        
        if context.strip():
            system_prompt = f"""You are a mathematics teacher. Respond in English.

            FORMAT RULE:
Do NOT output KaTeX configuration objects such as {{left:'', right:''}}.
Only output pure LaTeX inside $...$ or \[...\].

IMPORTANT: All mathematical formulas must be written in LaTeX format:
- For inline formulas: \\(formula\\)
- For displayed formulas: $$formula$$
- Use standard LaTeX notation

Example:
Function derivative: \\(f'(x) = \\lim_{{h \\to 0}} \\frac{{f(x+h)-f(x)}}{{h}}\\)
Integral: $$\\int_a^b f(x) dx$$

TEXTBOOK INFORMATION:
{context}

QUESTION: {question}

ANSWER (always use LaTeX for all mathematical expressions):
"""
        else:
            system_prompt = f"""You are a mathematics teacher. Respond clearly and in detail in English.

ALL mathematical formulas must be written in LaTeX:
- Inline: \\(formula\\)
- Displayed: $$formula$$

QUESTION: {question}

ANSWER:
"""
        
        api_key = st.secrets.get("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY"))
        if not api_key:
            return "❌ API key not configured. Add DEEPSEEK_API_KEY to Streamlit secrets."
        
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
                timeout=90
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"❌ API error ({response.status_code}): {response.text}"
                
        except Exception as e:
            return f"❌ Connection error: {str(e)}"

# ========== ИНТЕРФЕЙС STREAMLIT ==========
def render_math_answer(answer: str):
    """Displays answer with LaTeX support"""
    html = f"""
    <div class="math-content">
        {answer}
    </div>
    """
    return html

HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def main():
    # Инициализация темы
    if "dark_theme" not in st.session_state:
        st.session_state.dark_theme = True
    
    # Применяем тему сразу
    apply_custom_theme(st.session_state.dark_theme)
    
    # Кнопка переключения темы в сайдбаре
    with st.sidebar:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 🎨 Theme")
        with col2:
            theme_icon = "🌙" if st.session_state.dark_theme else "☀️"
            theme_label = "Dark" if st.session_state.dark_theme else "Light"
            if st.button(f"{theme_icon}", key="theme_toggle", help=f"Switch to {theme_label} theme"):
                st.session_state.dark_theme = not st.session_state.dark_theme
                st.rerun()
        
        st.markdown(f"**Current theme:** {'Dark' if st.session_state.dark_theme else 'Light'}")
        st.markdown("---")
    
    # Основной контент
    st.markdown('<h1 class="main-header">🎓 Math Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">AI math assistant based on your textbooks</p>', unsafe_allow_html=True)
    
    # Загружаем историю всегда (независимо от assistant)
    if "history" not in st.session_state:
        st.session_state.history = load_history()

    # Загружаем ассистента только один раз
    if "assistant" not in st.session_state:
        with st.spinner("🔄 Loading learning materials..."):
            st.session_state.assistant = MathAssistant("data")
    
    assistant = st.session_state.assistant
    
    # Сайдбар с контентом
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=100)
        st.markdown("### 📚 Loaded Subjects")
        
        if assistant.subjects:
            for subject_name, data in assistant.subjects.items():
                with st.container():
                    st.markdown(f"""
                    <div class="subject-card">
                    <strong>{data['config']['subject']}</strong><br>
                    📖 {len(data['config']['books'])} books<br>
                    🧩 {len(data['chunks'])} chunks
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Learning materials not loaded")
            st.info("""
            Create structure:
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
        st.markdown("### 💡 Example Questions")
        
        examples = [
            "What is a derivative?",
            "How to find matrix determinant?",
            "Explain L'Hopital's rule",
            "What are eigenvalues?"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}"):
                st.session_state.question = example
                st.rerun()
    
    # Основная область
    st.markdown("### 💭 Ask a Math Question")
    
    question = st.text_area(
        "Enter your question:",
        value=st.session_state.get("question", ""),
        placeholder="Example: 'What is a derivative?' or 'Explain Gauss method'",
        height=120,
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🎯 Get Answer", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("🔍 Searching in textbooks..."):
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

                    save_history(st.session_state.history)
                    
                    st.session_state.last_answer = answer
                    st.session_state.last_time = elapsed
                    st.rerun()
            else:
                st.warning("⚠️ Please enter a question")
    
    with col2:
        if st.button("🔄 New Question", use_container_width=True):
            if "last_answer" in st.session_state:
                del st.session_state.last_answer
            st.session_state.question = ""
            st.rerun()
    
    with col3:
        if st.button("📜 History", use_container_width=True):
            if "history" in st.session_state and st.session_state.history:
                st.markdown("### 📜 Question History")
                for i, item in enumerate(reversed(st.session_state.history[-5:])):
                    with st.expander(f"❓ {item['question'][:50]}..."):
                        st.markdown(f"**Time:** {item['time']:.1f} sec")
                        st.markdown("**Answer:**")
                        st.markdown(render_math_answer(item["answer"][:500] + ("..." if len(item["answer"]) > 500 else "")), unsafe_allow_html=True)
            else:
                st.info("📝 History is empty")
    
    if "last_answer" in st.session_state:
        st.markdown(f"### 📚 Answer ({st.session_state.get('last_time', 0):.1f} sec)")
        st.markdown("---")
        
        # Отображаем ответ с поддержкой LaTeX
        st.markdown(render_math_answer(st.session_state.last_answer), unsafe_allow_html=True)
        
        # Debug information
        with st.expander("📄 Raw answer text"):
            st.text(st.session_state.last_answer)
    
    with st.expander("ℹ️ About System"):
        st.markdown("""
        **How the system works:**
        1. 📚 Loads your textbooks (PDF → text)
        2. 🔍 Searches for relevant chunks by question
        3. 🤖 Sends context to DeepSeek AI
        4. 📝 Gets detailed answer
        
        **Supported topics:**
        - Mathematical analysis
        - Linear algebra
        - Differential equations
        
        **Requirements:**
        - DeepSeek API key (add to Streamlit secrets)
        - `data/` folder with textbook indexes
        
        **LaTeX support:**
        - All formulas automatically rendered using KaTeX
        - Use \\(formula\\) for inline formulas
        - Use $$formula$$ for displayed formulas
        """)
        
        if st.button("🧪 Test LaTeX rendering"):
            test_math = r"""
            **Mathematical formulas test:**
            
            Inline formula: \(E = mc^2\)
            
            Displayed formula:
            $$
            \int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
            $$
            
            Derivative: $$\frac{dy}{dx} = \lim_{\Delta x \to 0} \frac{f(x+\Delta x) - f(x)}{\Delta x}$$
            
            Matrix: $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$
            
            Sum: \(\sum_{i=1}^{n} i = \frac{n(n+1)}{2}\)
            """
            st.markdown(render_math_answer(test_math), unsafe_allow_html=True)

if __name__ == "__main__":
    main()