"""
AI Study Bot — Gemini (Google Gen AI) + Sentence-Transformers + FAISS
Full-featured Streamlit app (Summarize, Quiz Gen, Take Quiz, Explain, Teacher/Grill)

Requirements:
- Python 3.9+
- pip install -r requirements.txt

Suggested requirements.txt contents:
google-genai
streamlit
sentence-transformers
faiss-cpu
PyPDF2
numpy

Environment variables:
- GEMINI_API_KEY (recommended) or GOOGLE_API_KEY

Run:
1) create & activate venv
2) pip install -r requirements.txt
3) export GEMINI_API_KEY="your_key_here"  (or set in PowerShell)
4) streamlit run ai-study-bot-gemini.py

NOTES / LIMITATIONS:
- Embeddings are computed locally using sentence-transformers (all-MiniLM-L6-v2). That means CPU time but no external embedding costs.
- Gemini quota/limits apply. Keep retrieval contexts small (3-6 chunks) to avoid large token usage.
- This app is a starter. Secure the key for deployment.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from PyPDF2 import PdfReader
import numpy as np

# Google GenAI SDK
from google import genai

# Embeddings & FAISS
from sentence_transformers import SentenceTransformer
import faiss

# ---------------- CONFIG ----------------
DATA_DIR = Path("./ai_study_bot_gemini_data")
DATA_DIR.mkdir(exist_ok=True)
INDEX_PATH = DATA_DIR / "faiss.idx"
CHUNKS_PATH = DATA_DIR / "chunks.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 produces 384-d vectors
MODEL_NAME = "gemini-1.5"  # default; change if you have access to flash/pro

# ---------------- HELPERS ----------------

def ensure_genai_client():
    # The google genai Client picks api key from GEMINI_API_KEY or GOOGLE_API_KEY
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        st.error("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set. Set it and re-run.")
        st.stop()
    # Optional: return a client if you want
    return genai.Client(api_key=key)


@st.cache_data
def load_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        chunk = text[i:i+chunk_size]
        chunks.append(chunk.strip())
        i += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]

def build_faiss_index(chunks: List[str], embedder: SentenceTransformer):
    vectors = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    # save
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    return index

def load_faiss_index():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def retrieve(query: str, index, chunks: List[str], embedder: SentenceTransformer, k: int = 3) -> List[str]:
    q_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec, k)
    results = []
    for idx in I[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
    return results

# Gemini calls

def gemini_generate(prompt: str, context: str = None) -> str:
    payload = prompt if not context else f"Context:\n{context}\n\nInstruction:\n{prompt}"
    client = genai.Client()  # API key picked from env

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[payload]  # just a list of strings
    )

    # Extract text (SDK returns response.result)
    try:
        text = response.result[0].content[0].text
    except Exception:
        text = str(response)
    return text



# Grading helper: ask Gemini to return JSON {score:0-10, justification:..., tips:[..]}
def gemini_grade(ideal: str, student: str) -> Dict[str, Any]:
    system = (
        "You are an objective grader. Rate the student's answer against the ideal answer on a 0-10 scale. "
        "Return VALID JSON only with keys: score (int 0-10), justification (one sentence), tips (array of 1-3 short tips)."
    )
    prompt = f"Ideal Answer:\n{ideal}\n\nStudent Answer:\n{student}\n\n{system}"
    out = gemini_generate(prompt, max_output_tokens=300, temperature=0.0)
    # Try parse JSON from output
    try:
        parsed = json.loads(out)
    except Exception:
        # Attempt to extract JSON substring
        import re
        m = re.search(r"\{[\s\S]*\}", out)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = {"score": None, "justification": out.strip(), "tips": []}
        else:
            parsed = {"score": None, "justification": out.strip(), "tips": []}
    return parsed

# ---------------- STREAMLIT APP ----------------

st.set_page_config(page_title="AI Study Bot — Gemini", layout="wide")
st.title("📚 AI Study Bot — Gemini Edition")

with st.sidebar:
    st.header("Notes / Index")
    uploaded = st.file_uploader("Upload a PDF (single file)", type=["pdf"], accept_multiple_files=False)
    if uploaded:
        text = load_text_from_pdf(uploaded)
        st.write(f"Loaded {len(text)} characters")
        chunk_size = st.number_input("Chunk size", min_value=400, max_value=2000, value=800)
        overlap = st.number_input("Chunk overlap", min_value=50, max_value=500, value=150)
        if st.button("Build local index from uploaded notes"):
            with st.spinner("Chunking & embedding (this may take a while on CPU)..."):
                embedder = get_embedder()
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                index = build_faiss_index(chunks, embedder)
                st.success(f"Built index with {len(chunks)} chunks and saved locally.")

    elif INDEX_PATH.exists():
        if st.button("Load existing local index"):
            idx, ch = load_faiss_index()
            if idx is None:
                st.error("Index not found on disk. Build one from uploaded notes.")
            else:
                st.success(f"Loaded index with {len(ch)} chunks.")
    else:
        st.info("Upload a PDF and build an index to enable retrieval features.")

# Ensure genai client is configured when needed
ensure_genai_client()

cols = st.columns([2,3])

with cols[0]:
    st.subheader("Modes")
    mode = st.radio("Choose mode:", ["Summarize", "Quiz Generator", "Take Quiz", "Explain (Feynman)", "Teacher (Grill)"])
    st.write("---")
    st.write("Data files saved to: ", str(DATA_DIR))

with cols[1]:
    st.subheader("Workspace")
    index, chunks = load_faiss_index()
    embedder = get_embedder()

    if mode == "Summarize":
        st.markdown("### Summarize Notes")
        brief = st.checkbox("Brief summary (high-level)", value=True)
        if st.button("Summarize"):
            if index is None:
                st.error("No index found. Upload notes and build index first.")
            else:
                # retrieve top chunks for summary
                ctx = "\n---\n".join(retrieve("summarize the notes", index, chunks, embedder, k=6))
                instr = "Summarize the provided notes into concise bullet points (8-12 bullets)." if brief else "Summarize the provided notes in a detailed multi-level format."
                with st.spinner("Generating summary from Gemini..."):
                    out = gemini_generate(instr, context=ctx)
                    st.markdown(out)

    elif mode == "Quiz Generator":
        st.markdown("### Generate Quiz")
        n_mcq = st.slider("MCQs", 1, 10, 5)
        n_short = st.slider("Short-answer Qs", 0, 10, 3)
        if st.button("Generate Quiz"):
            if index is None:
                st.error("No index found. Upload notes and build index first.")
            else:
                ctx = "\n---\n".join(retrieve("generate quiz questions", index, chunks, embedder, k=6))
                prompt = (
                    f"Based on the context, create {n_mcq} multiple-choice questions and {n_short} short-answer questions. "
                    "Return JSON with keys 'mcqs' (each: question, choices, answer, explanation) and 'shorts' (each: question, answer, explanation)."
                )
                with st.spinner("Asking Gemini to make a quiz..."):
                    raw = gemini_generate(prompt, context=ctx, max_output_tokens=1000, temperature=0.3)
                    # try parse JSON
                    try:
                        quiz = json.loads(raw)
                    except Exception:
                        # attempt to extract JSON
                        import re
                        m = re.search(r"\{[\s\S]*\}", raw)
                        if m:
                            try:
                                quiz = json.loads(m.group(0))
                            except Exception:
                                quiz = {"mcqs": [], "shorts": [], "raw": raw}
                        else:
                            quiz = {"mcqs": [], "shorts": [], "raw": raw}
                    # save quiz locally
                    qname = f"quiz_{int(time.time())}"
                    with open(DATA_DIR / f"{qname}.json", "w", encoding="utf-8") as f:
                        json.dump(quiz, f, ensure_ascii=False, indent=2)
                    st.success("Quiz generated and saved")
                    st.json(quiz)

    elif mode == "Take Quiz":
        st.markdown("### Take a Saved Quiz")
        quizzes = list(DATA_DIR.glob("quiz_*.json"))
        if not quizzes:
            st.info("No generated quizzes found. Create one in 'Quiz Generator' mode first.")
        else:
            picked = st.selectbox("Choose quiz file", [q.name for q in quizzes])
            with open(DATA_DIR / picked, "r", encoding="utf-8") as f:
                quiz = json.load(f)
            st.write("#### MCQs")
            user_answers = {}
            for i, mcq in enumerate(quiz.get("mcqs", [])):
                st.write(f"**Q{i+1}. {mcq.get('question','')}**")
                choices = mcq.get("choices", [])
                key = f"mcq_{i}"
                if choices:
                    user_answers[key] = st.radio(f"Choose (Q{i+1})", choices, key=key)
                else:
                    user_answers[key] = st.text_input(f"Answer (Q{i+1})", key=key)

            st.write("#### Short-answer questions")
            for i, sq in enumerate(quiz.get("shorts", [])):
                key = f"short_{i}"
                user_answers[key] = st.text_area(f"Q{i+1+len(quiz.get('mcqs',[]))}. {sq.get('question','')}", key=key)

            if st.button("Submit Quiz"):
                with st.spinner("Grading..."):
                    total = 0.0
                    details = []
                    # MCQ scoring
                    for i, mcq in enumerate(quiz.get("mcqs", [])):
                        correct = mcq.get("answer")
                        user_choice = user_answers.get(f"mcq_{i}")
                        is_correct = False
                        if isinstance(user_choice, str) and correct:
                            is_correct = user_choice.strip().lower() == correct.strip().lower()
                        details.append({"q": mcq.get('question'), "correct": correct, "user": user_choice, "is_correct": is_correct})
                        if is_correct:
                            total += 1.0

                    # Short answers: grade with Gemini
                    for i, sq in enumerate(quiz.get("shorts", [])):
                        ideal = sq.get("answer", "")
                        student = user_answers.get(f"short_{i}", "")
                        if student.strip() == "":
                            grade = {"score": 0, "justification": "No answer provided", "tips": []}
                        else:
                            grade = gemini_grade(ideal, student)
                        # Interpret score
                        s = grade.get("score")
                        if isinstance(s, (int, float)):
                            total += float(s) / 10.0
                        details.append({"q": sq.get('question'), "ideal": ideal, "user": student, "grade": grade})

                    max_possible = len(quiz.get("mcqs", [])) + len(quiz.get("shorts", []))
                    percent = (total / max_possible) * 100 if max_possible else 0
                    st.success(f"You scored: {percent:.1f}%")
                    st.json(details)
                    # Save to history
                    hist = DATA_DIR / "history.json"
                    h = []
                    if hist.exists():
                        h = json.loads(hist.read_text())
                    h.append({"quiz": picked, "percent": percent, "details": details, "ts": int(time.time())})
                    hist.write_text(json.dumps(h, indent=2))

    elif mode == "Explain (Feynman)":
        st.markdown("### Explain (Feynman) Mode")
        concept = st.text_input("Enter concept to explain:")
        if st.button("Explain"):
            if index is None:
                st.error("No index found. Upload notes and build index first.")
            else:
                ctx = "\n---\n".join(retrieve(concept, index, chunks, embedder, k=4))
                instr = f"Explain '{concept}' as if to a smart 10-year-old. Use an analogy and give 3 one-sentence examples. If concept not found in context, still provide a clear explanation."
                out = gemini_generate(instr, context=ctx, temperature=0.25)
                st.markdown(out)

    elif mode == "Teacher (Grill)":
        st.markdown("### Teacher / Grill Mode")
        n_q = st.slider("Number of questions to generate", 1, 10, 5)
        if st.button("Generate Questions"):
            if index is None:
                st.error("No index found. Upload notes and build index first.")
            else:
                ctx = "\n---\n".join(retrieve("important questions", index, chunks, embedder, k=6))
                instr = f"Generate {n_q} tough open-ended questions from the context. Number them."
                raw = gemini_generate(instr, context=ctx, temperature=0.4)
                # simple parse into list
                qs = []
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line[0].isdigit():
                        q = line.split('.',1)[1].strip() if '.' in line else line
                        qs.append({"question": q})
                    else:
                        if qs:
                            qs[-1]["question"] += " " + line
                if not qs:
                    qs = [{"question": raw}]
                st.session_state["teacher_questions"] = qs
                st.success("Questions generated. Answer them below and click 'Grade my answers'.")

        if st.session_state.get("teacher_questions"):
            for i, q in enumerate(st.session_state["teacher_questions"]):
                st.write(f"**Q{i+1}. {q.get('question')}**")
                st.text_area(f"Your answer (Q{i+1})", key=f"teach_ans_{i}")
            if st.button("Grade my answers"):
                results = []
                for i, q in enumerate(st.session_state["teacher_questions"]):
                    ideal_instr = f"Provide an ideal 3-5 sentence answer for: {q.get('question')}"
                    ideal = gemini_generate(ideal_instr, temperature=0.2)
                    student = st.session_state.get(f"teach_ans_{i}", "")
                    grade = gemini_grade(ideal, student)
                    results.append({"question": q.get('question'), "ideal": ideal, "student": student, "grade": grade})
                st.json(results)
                hist = DATA_DIR / "history.json"
                h = []
                if hist.exists():
                    h = json.loads(hist.read_text())
                h.append({"ts": int(time.time()), "teacher_results": results})
                hist.write_text(json.dumps(h, indent=2))

# footer
st.write("---")
st.caption("This app runs embeddings locally (sentence-transformers) and calls Gemini for text generation/grading. Keep your GEMINI_API_KEY safe.")
