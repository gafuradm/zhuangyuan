import streamlit as st
import os
import json
import requests
import numpy as np
import hnswlib
from typing import List
import time
import hashlib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import re

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="Mathematics Assistant",
    page_icon="📚",
    layout="wide"
)

# Load KaTeX at the very beginning
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

# CSS styles
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
    /* Styles for mathematical content */
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
</style>
""", unsafe_allow_html=True)

# ========== EMBEDDING MODEL ==========
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

# ========== MAIN CLASS ==========
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
            st.warning("⚠️ No subjects in the data/ folder")
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
        """Determines the subject of the question"""
        question_lower = question.lower()
        subject_keywords = {
            "matan": ["mathematical analysis", "calculus", "differential", "integral", 
                     "limit", "series", "function", "derivative", "differentiation"],
            "linalg": ["linear", "matrix", "vector", "determinant", 
                      "eigen", "linear space", "linear algebra"]
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
            system_prompt = f"""
You are a mathematics teacher. Answer ONLY in clean LaTeX.

STRICT RULES (must obey):
1. Never output stray characters
2. Every opening.
3. No broken fragments.
4. All formulas must be inside:
   - Inline: \\( ... \\)
   - Displayed: $$ ... $$

5. Russian or English text must be outside math mode.
   Example:
   Пусть функция \\(f(x)\\) непрерывна…

6. NO KaTeX configuration objects like {{left:'', right:''}}.

QUESTION:
{question}

ANSWER ONLY IN CLEAN PROPER LaTeX:

INFORMATION FROM TEXTBOOKS:
{context}

QUESTION: {question}

ANSWER (always use LaTeX for all mathematical expressions):
"""
        else:
            system_prompt = f"""You are a mathematics teacher. Answer clearly and in detail in English.

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
                return f"❌ API Error ({response.status_code}): {response.text}"
                
        except Exception as e:
            return f"❌ Connection error: {str(e)}"

# ========== STREAMLIT INTERFACE ==========
def render_math_answer(answer: str):
    """Displays answer with LaTeX support"""
    # Wrap answer in div with styling class
    
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

def create_pdf(answer: str) -> bytes:
    # Создаем буфер для PDF
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    
    # Устанавливаем позицию для текста
    y_position = 750
    line_height = 14
    
    # Разбиваем текст на строки
    lines = answer.split("\n")
    
    # Добавляем заголовок
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y_position, "Mathematics Assistant - Answer")
    y_position -= 30
    
    # Основной текст
    pdf.setFont("Helvetica", 12)
    
    for line in lines:
        # Если текст слишком длинный, разбиваем на несколько строк
        if len(line) > 100:
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line + " " + word) <= 100:
                    current_line += " " + word if current_line else word
                else:
                    if y_position < 50:  # Новая страница
                        pdf.showPage()
                        pdf.setFont("Helvetica", 12)
                        y_position = 750
                    pdf.drawString(40, y_position, current_line)
                    y_position -= line_height
                    current_line = word
            if current_line:
                if y_position < 50:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 12)
                    y_position = 750
                pdf.drawString(40, y_position, current_line)
                y_position -= line_height
        else:
            if y_position < 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 12)
                y_position = 750
            pdf.drawString(40, y_position, line)
            y_position -= line_height
    
    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

def parse_latex_tasks(raw: str):
    tasks = []

    # 1) Убираем переносы в \text{...} чтобы не ломало парсер
    raw = re.sub(r"\\text\{([^}]*)\n([^}]*)\}", r"\\text{\1 \2}", raw)

    # --------- ПАТТЕРН 1: \[  \] -----------
    blocks = re.findall(r"\\\[(.*?)\\\]", raw, flags=re.S)
    for b in blocks:
        m = re.search(r"ЗАДАЧА\s*\d+[:\.]?\s*(.*)", b, flags=re.I)
        if m:
            tasks.append(m.group(1).strip())

    # --------- ПАТТЕРН 2: $$  $$ -----------
    blocks = re.findall(r"\$\$(.*?)\$\$", raw, flags=re.S)
    for b in blocks:
        m = re.search(r"ЗАДАЧА\s*\d+[:\.]?\s*(.*)", b, flags=re.I)
        if m:
            tasks.append(m.group(1).strip())

    # --------- ПАТТЕРН 3: \(  \) ------------
    blocks = re.findall(r"\\\((.*?)\\\)", raw, flags=re.S)
    for b in blocks:
        m = re.search(r"ЗАДАЧА\s*\d+[:\.]?\s*(.*)", b, flags=re.I)
        if m:
            tasks.append(m.group(1).strip())

    # --------- ПАТТЕРН 4: Просто текст ------
    lines = raw.splitlines()
    for line in lines:
        m = re.match(r"\s*ЗАДАЧА\s*\d+[:\.]?\s*(.*)", line, flags=re.I)
        if m:
            tasks.append(m.group(1).strip())

    return tasks

def generate_test(topic: str, count: int, difficulty: str, style: str, api_key: str):
    prompt = f"""
Ты — генератор экзаменационных задач.

Сформируй {count} задач по теме "{topic}".
Сложность: {difficulty}.
Стиль: {style}.

❗ Выводи СТРОГО в формате LaTeX:
Каждая задача должна быть оформлена так:

\\[
\\text{{ЗАДАЧА 1: }} <текст задачи в одной строке>
\\]

Только задачи. Без решений. Без лишнего текста.
"""

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Ты — математический экзаменатор. Всегда выводи в чистом LaTeX."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=60
    )

    return response.json()["choices"][0]["message"]["content"]


def check_answers(tasks, user_answers, api_key: str):
    prompt = "Ты — строгий экзаменатор. Проверь ответы студента.\n\n"

    for i, task in enumerate(tasks, 1):
        prompt += f"""
ЗАДАЧА {i}: {task}

Ответ студента: {user_answers.get(i, '---')}
---
"""

    prompt += """
Проанализируй КАЖДУЮ задачу.
Выводи строго в LaTeX в таком формате:

\\[
\\text{Задача 1: } \checkmark \text{ или } \times
\\]

\\[
\\text{Правильный ответ: } <формула>
\\]

\\[
\\text{Объяснение: } <1–2 строки>
\\]

В конце выведи:

\\[
\\text{ИТОГОВЫЙ БАЛЛ: } <число>/<кол-во задач>
\\]
"""

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Всегда выводи только LaTeX. Никакого текста вне формул."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=120
    )

    return response.json()["choices"][0]["message"]["content"]

def main():
    st.markdown('<h1 class="main-header">🎓 Mathematics Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI mathematics assistant based on your textbooks</p>', unsafe_allow_html=True)
    
    # Always load history (independent of assistant)
    if "history" not in st.session_state:
        st.session_state.history = load_history()

    page = st.sidebar.selectbox("📂 Pages", ["Chat", "Test Maker", "History"])

    # Load assistant only once
    if "assistant" not in st.session_state:
        with st.spinner("🔄 Loading learning materials..."):
            st.session_state.assistant = MathAssistant("data")

    
    assistant = st.session_state.assistant
    
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
            "Explain L'Hôpital's rule",
            "What are eigenvalues?"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}"):
                st.session_state.question = example
                st.rerun()
    
    st.markdown("### 💭 Ask a Mathematics Question")
    
    question = st.text_area(
        "Enter your question:",
        value=st.session_state.get("question", ""),
        placeholder="Example: 'What is a derivative?' or 'Explain Gauss elimination method'",
        height=120,
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🎯 Get Answer", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("🔍 Searching information in textbooks..."):
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
                st.info("📝 Question history is empty")
    
    if "last_answer" in st.session_state:
        st.markdown(f"### 📚 Answer ({st.session_state.get('last_time', 0):.1f} sec)")
        st.markdown("---")
        
        # Display answer with LaTeX support
        st.markdown(render_math_answer(st.session_state.last_answer), unsafe_allow_html=True)
        
        # PDF download button
        pdf_bytes = create_pdf(st.session_state.last_answer)
        st.download_button(
            label="📄 Download answer as PDF",
            data=pdf_bytes,
            file_name="answer.pdf",
            mime="application/pdf"
        )

        # Debug information (can be hidden)
        with st.expander("📄 Raw answer text"):
            st.text(st.session_state.last_answer)
    elif page == "Test Maker":
        api_key = st.secrets.get("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY"))
        if not api_key:
            st.error("❌ No API key found.")
            return

        st.title("📝 Test Maker — генератор экзаменов")

        # Состояния
        if "test_tasks" not in st.session_state:
            st.session_state.test_tasks = None

        # Если тест еще не создан
        if st.session_state.test_tasks is None:
            st.subheader("Создать тест")

            topic = st.text_input("📌 Тема", "Интегралы")
            count = st.number_input("🔢 Кол-во задач", 1, 30, 10)
            difficulty = st.selectbox("🔥 Сложность", ["Легко", "Средне", "Сложно", "Олимпиада"])
            style = st.selectbox("📖 Стиль задач", ["Авторские", "Из учебников", "Смешанные"])

            if st.button("🎯 Сгенерировать тест"):
                with st.spinner("ИИ генерирует задачи..."):
                    raw = generate_test(topic, count, difficulty, style, api_key)

                # Парсим задачи
                tasks = parse_latex_tasks(raw)

                if not tasks:
                    st.error("❌ Не удалось распарсить задачи.")
                else:
                    st.session_state.test_tasks = tasks
                    st.rerun()

        # Если тест уже создан
        else:
            st.subheader("📘 Ваш тест")

            tasks = st.session_state.test_tasks
            user_answers = {}

            for i, task in enumerate(tasks, 1):
                st.markdown(f"### 🧩 Задача {i}")
                st.markdown(task)
                user_answers[i] = st.text_area(f"Ответ {i}", key=f"answer_{i}")

            if st.button("✅ Проверить ответы"):
                with st.spinner("ИИ проверяет..."):
                    result = check_answers(tasks, user_answers, api_key)

                st.markdown("### 📊 Результаты")
                st.markdown(render_math_answer(result), unsafe_allow_html=True)

            if st.button("🔄 Новый тест"):
                st.session_state.test_tasks = None
                st.rerun()
    
    with st.expander("ℹ️ About the System"):
        st.markdown("""
        **How the system works:**
        1. 📚 Loads your textbooks (PDF → text)
        2. 🔍 Searches for relevant chunks based on the question
        3. 🤖 Sends context to DeepSeek AI
        4. 📝 Receives detailed answer
        
        **Supported topics:**
        - Mathematical Analysis
        - Linear Algebra
        - Differential Equations
        
        **Requirements:**
        - DeepSeek API key (add to Streamlit secrets)
        - `data/` folder with textbook indexes
        
        **LaTeX support:**
        - All formulas are automatically rendered using KaTeX
        - Use \\(formula\\) for inline formulas
        - Use $$formula$$ for displayed formulas
        """)
        
        if st.button("🧪 Test LaTeX Rendering"):
            test_math = r"""
            **Mathematical Formulas Test:**
            
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