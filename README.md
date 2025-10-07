# AI-CHATBOT

A full-featured AI study assistant that uses Google Gemini (Gen AI) and sentence-transformers + FAISS to process notes, generate quizzes, summarize content, and explain concepts. Built in Python with a Streamlit interface.

Features

Upload PDF notes: Automatically chunk and embed your notes for fast retrieval.

Summarize: Generate concise or detailed summaries from your notes.

Quiz Generator: Create multiple-choice and short-answer quizzes automatically.

Take Quiz: Solve quizzes and get automated grading, including AI-assisted grading for short answers.

Explain (Feynman): Enter a concept and get a simplified, easy-to-understand explanation with examples.

Teacher / Grill Mode: Generate tough questions from your notes and get AI-generated ideal answers and grading.

Requirements

Python 3.9+

Install dependencies:

pip install -r requirements.txt


Suggested requirements.txt:

google-genai
streamlit
sentence-transformers
faiss-cpu
PyPDF2
numpy

Setup

Clone the repository:

git clone <your_repo_url>
cd <repo_folder>


Create and activate a virtual environment:

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Set your Google Gemini API key:

# Windows (PowerShell)
setx GEMINI_API_KEY "your_api_key_here"
# macOS/Linux
export GEMINI_API_KEY="your_api_key_here"

Usage

Run the Streamlit app:

streamlit run ai-study-bot-gemini.py


Upload your PDF notes in the sidebar and build a local index.

Choose a mode (Summarize, Quiz Generator, Take Quiz, Explain, Teacher/Grill) to start interacting with your notes.

How It Works

Embeddings & Indexing: Notes are split into overlapping chunks, converted into embeddings using sentence-transformers, and indexed with FAISS for fast retrieval.

Contextual AI Generation: Gemini receives the most relevant chunks and generates summaries, quizzes, or explanations.

Quiz Grading: Multiple-choice questions are automatically scored. Short-answer questions are graded using AI for detailed feedback.

Notes & Limitations

Embeddings are computed locally (CPU usage may be high).

Gemini API quota applies; keep retrieval contexts small (3–6 chunks).

Only works with models available on your Gemini API key (gemini-1.0, gemini-1.5-large, etc.).

Project Structure
ai-study-bot-gemini/
├─ ai-study-bot-gemini.py  # main Streamlit app
├─ requirements.txt
├─ ai_study_bot_gemini_data/  # stores FAISS index & generated quiz/history files
└─ README.md

Future Improvements

Add support for image/PDF diagrams in notes.

Enable multi-document indexing.

Add authentication for secure deployments.

Author

RishiKhanth

Email: rishi10xcr7@gmail.com

GitHub: Riisshi