Resume Screening Agent

An AI-powered Resume Screening Agent that parses resumes & job descriptions, extracts skills and experience, computes semantic similarity using local embeddings, and produces ranked candidate output with a downloadable CSV.
Designed to work fully offline using Sentence-Transformers — with optional future support for OpenAI/LLMs.

🚀 Features:
✔ Upload Job Description (paste or file upload)
✔ Upload multiple resumes (PDF / DOCX / TXT)
✔ Automatic parsing of:
    Skills
    Experience level
    Education indicators
    Projects
    Contact info (email/phone)

✔ Semantic similarity using local embeddings
    all-MiniLM-L6-v2 (fast & accurate offline model)

✔ Weighted scoring engine
    Industry-standard weights:
        Skills match – 40%
        Experience – 30%
        Title/keyword match – 15%
        Education – 10%
        Projects – 5%

✔ Human-readable summary per candidate
    Generated using a local rule-based finalizer (no LLM required).

✔ Export results
    CSV download
    Includes scores, rationale, summary & key fields

🏗 Architecture Diagram:
The architecture diagram used in the project is stored at:
    architecture/architecture_diagram.png

📁 Folder Structure:
resume-screener/
│
├── app/
│   ├── main.py                  # Streamlit UI
│   ├── processor.py             # Parsing, embeddings, scoring, summaries
│   ├── utils.py                 # Resume/JD text extraction
│   ├── model_config.py          # Weights for scoring
│   └── architecture_page.py     # Page showing architecture diagram
│
├── architecture/
│   └── architecture_diagram.png
│
├── docs/
│   └── demo_instructions.md
│
├── requirements.txt
├── README.md
├── .env.example
└── .env (ignored)

⚙️ Installation & Running (Offline)
1) Create virtual environment
py -m venv venv
.\venv\Scripts\Activate.ps1

2) Install dependencies
pip install -r requirements.txt
pip install -U sentence-transformers scikit-learn

3) Enable offline mode (local embeddings)
Inside .env add:
USE_LOCAL_EMB=true

4) Run the app
streamlit run app/main.py

🧠 How Scoring Works:
The scoring engine computes a weighted final score between 0–1.

1. Skills Similarity (40%)
Local embedding vectors from resume & JD
Cosine similarity → normalized score

2. Experience Score (30%)
Extract candidate experience from text
Extract minimum required experience from JD

3. Title / Keyword Match (15%)
Checks if job title keywords appear in resume text

4. Education (10%)
Checks presence of BTech/BSc/MTech/Masters/PhD keywords

5. Projects (5%)
Checks for "projects", "research", "publications" sections

Human Summary:
    A concise rule-based summary is created:
    Highlights strong or weak skill match
    Mentions experience fit
    Lists top extracted skills
    Provides a quick recommendation-style overview

🧩 Tech Stack:
1. Backend
    Python 3
    Sentence-Transformers
    scikit-learn
    regex-based parsers
    Optional: OpenAI API (future)

2. Frontend
    Streamlit
    Custom CSS for layout & improved UX

3. Storage
    Local filesystem
    Optional: ChromaDB for vector index (planned)

💡 Potential Improvements:
🚀 Upgrade to cloud LLM
    Improve summaries using GPT-4 or Claude
    Generate richer rationales (key phrases, strengths, weaknesses)

🚀 Add storage/persistence
    Save candidate profiles in a persistent vector DB (Chroma/Pinecone)

🚀 Add skill extraction model
    Replace rule-based extraction with spaCy / transformer-based NER

🚀 Add multi-role JD comparison
    Support uploading multiple job descriptions
    Rank candidates for several roles simultaneously

🚀 Add recruiter dashboard
    Pagination
    Download PDF of all candidates
    Candidate shortlisting workflows

🚀 Deployment
    Streamlit Cloud
    HuggingFace Spaces
    Docker image for easy deployment

🎯 Why Local Embeddings?
    Works offline
    No quota/billing limits
    Fast inference
    Judges can run without external API keys
    Architecture still supports quick swapping to OpenAI embeddings
