import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from app.processor import parse_resume_file, parse_job_description, score_candidate, generate_human_summary

st.set_page_config(page_title="Resume Screening Agent", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] .main .block-container { padding-top: 0.6rem; padding-left: 1rem; padding-right: 1rem; }
    html, body, [data-testid="stAppViewContainer"] { height:100%; margin:0; padding:0; overflow:hidden; }
    [data-testid="stAppViewContainer"] > .main { min-height:100vh; height:100vh; overflow:hidden; }
    [data-testid="stSidebar"] { overflow:auto; max-height:100vh; }
    .no-page-scroll-box { height: calc(100vh - 5rem); overflow:auto; padding-right: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Resume Screening Agent — Demo")
st.write("Upload a job description and multiple resumes (PDF/DOCX/TXT). The agent will parse, score and rank candidates.")

with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Paste or upload the Job Description.  
    2. Upload multiple resumes (PDF/DOCX/TXT).  
    3. Click **Run Screening**.  
    4. Download CSV of ranked candidates for submission.
    """)
    st.markdown("---")
    st.write("Settings")
    max_res = st.number_input("Max resumes to process (demo)", min_value=1, max_value=200, value=50, step=5)

st.header("1) Job Description (paste or upload)")
col1, col2 = st.columns([2,1])
with col1:
    jd_text = st.text_area("Paste JD text here", placeholder="Paste the complete job description here...", height=160)
with col2:
    jd_file = st.file_uploader("Or upload a JD file (txt/pdf/docx)", type=['txt','pdf','docx'])
    if jd_file is not None:
        tmpjd = os.path.join("tmp_uploads", f"jd_{jd_file.name}")
        os.makedirs("tmp_uploads", exist_ok=True)
        with open(tmpjd, "wb") as out:
            out.write(jd_file.getbuffer())
        with open(tmpjd, "r", encoding="utf-8", errors="ignore") as f:
            try:
                jd_text = f.read()
            except Exception:
                jd_text = ""

st.markdown("---")
st.header("2) Upload Resumes (PDF / DOCX / TXT)")
uploaded_files = st.file_uploader("Upload resumes", accept_multiple_files=True, type=['pdf','docx','doc','txt'])
run_btn = st.button("Run Screening")

if run_btn:
    if not jd_text or not uploaded_files:
        st.error("Paste a JD and upload at least one resume.")
    else:
        st.info("Parsing job description...")
        parsed_jd = parse_job_description(jd_text)

        st.info(f"Processing {min(len(uploaded_files), max_res)} resumes...")
        results = []
        progress_bar = st.progress(0)
        processed = 0

        for f in uploaded_files[:max_res]:
            tmp_path = os.path.join("tmp_uploads", f.name)
            os.makedirs("tmp_uploads", exist_ok=True)
            with open(tmp_path, "wb") as out:
                out.write(f.getbuffer())
            parsed = parse_resume_file(tmp_path)
            scored = score_candidate(parsed, parsed_jd)

            scored["email"] = parsed.get("contact", {}).get("email")
            scored["phone"] = parsed.get("contact", {}).get("phone")
            scored["years_experience"] = parsed.get("years_experience", 0)
            scored["top_skills"] = ", ".join(parsed.get("skills", [])[:8])
            scored["filename"] = parsed.get("filename")

            scored["summary"] = generate_human_summary(parsed, scored)
            results.append(scored)
            processed += 1
            progress_bar.progress(processed / min(len(uploaded_files), max_res))

        # Sort and present
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        st.success("Screening complete")

        # Make dataframe for download
        rows = []
        for r in results_sorted:
            rows.append({
                "name": r.get("name") or r.get("filename"),
                "filename": r.get("filename"),
                "score": round(r.get("score",0), 4),
                "skills_score": round(r["component_scores"].get("skills",0), 4),
                "experience_score": round(r["component_scores"].get("experience",0), 4),
                "title_score": round(r["component_scores"].get("title_keywords",0), 4),
                "education_score": round(r["component_scores"].get("education",0), 4),
                "projects_score": round(r["component_scores"].get("projects",0), 4),
                "years_experience": r.get("years_experience"),
                "email": r.get("email"),
                "phone": r.get("phone"),
                "top_skills": r.get("top_skills"),
                "summary": r.get("summary")
            })
        df = pd.DataFrame(rows)

        # Show ranked table
        st.header("Ranked Candidates")
        st.dataframe(df.style.format({"score":"{:.3f}"}), height=320)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download results as CSV", data=csv, file_name="screening_results.csv", mime="text/csv")

        st.header("Candidate details")
        for r in results_sorted:
            label = f"{r.get('name') or r.get('filename')} — Score: {r.get('score'):.3f}"
            with st.expander(label, expanded=False):
                st.write("**Summary:**", r.get("summary"))
                st.write("**Component Scores:**")
                st.json(r.get("component_scores"))
                st.write("**Rationale:**")
                st.write(r.get("rationale"))
                st.write("**Highlights:**")
                for h in r.get("highlights", [])[:6]:
                    st.write("-", h)
                st.write("---")
                st.write("**Top skills (extracted):**", r.get("top_skills"))
                st.write("**Contact:**", r.get("email"), r.get("phone"))

else:
    st.info("When you press 'Run Screening', uploaded resumes will be parsed, embedded and ranked. Use the sidebar to change settings.")
