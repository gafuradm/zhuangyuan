import streamlit as st
import os
import json
import requests
import numpy as np
import hnswlib
from typing import List
import time
import hashlib

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
            system_prompt = f"""You are a mathematics teacher. Answer in English.

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

def main():
    st.markdown('<h1 class="main-header">🎓 Mathematics Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI mathematics assistant based on your textbooks</p>', unsafe_allow_html=True)
    
    # Always load history (independent of assistant)
    if "history" not in st.session_state:
        st.session_state.history = load_history()

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
        
        # Debug information (can be hidden)
        with st.expander("📄 Raw answer text"):
            st.text(st.session_state.last_answer)
    
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