# app/processor.py
import os
import re
import math
import json
from collections import Counter
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

USE_LOCAL_EMB = os.getenv("USE_LOCAL_EMB", "false").lower() in ("1","true","yes")
USE_OPENAI = not USE_LOCAL_EMB and bool(os.getenv("OPENAI_API_KEY"))

if USE_LOCAL_EMB:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer("all-MiniLM-L6-v2")
else:
    try:
        from openai import OpenAI
        openai_client = OpenAI()
    except Exception:
        openai_client = None

from app.utils import extract_text_from_file
from app.model_config import WEIGHTS

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{6,}\d)")
YEAR_RE = re.compile(r"(\d{4})")

COMMON_EDU_KEYWORDS = ["bachelor", "master", "phd", "b.sc", "btech", "mtech", "bachelor of", "master of", "mba", "doctorate"]
PROJECT_SECTIONS = ["project", "projects", "research", "publications"]

def extract_contact(text):
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    return {"email": email.group(0) if email else None, "phone": phone.group(0) if phone else None}

def extract_name(text):
    for line in text.splitlines():
        s = line.strip()
        if 3 > len(s) > 2 and all(not kw in s.lower() for kw in ("resume","cv","curriculum","address","phone","email")):
            return s
    return None

def extract_years_experience(text):
    ranges = re.findall(r"(\d{4})\s*[-–]\s*(\d{4}|Present|present)", text)
    years = []
    for start, end in ranges:
        try:
            s = int(start)
            e = int(end) if end.isdigit() else 2025
            if e >= s:
                years.append(e - s)
        except:
            continue
    if years:
        return max(sum(years), 0)
    m = re.search(r"(\d+)\+?\s+years?", text.lower())
    if m:
        return int(m.group(1))
    return 0

def extract_skills(text, top_k=80):
    skills = []
    for line in text.splitlines():
        if "skill" in line.lower() or "technology" in line.lower() or "stack" in line.lower():
            parts = re.split(r"[,\|/;•\t]", line)
            for p in parts:
                p = p.strip()
                if 2 <= len(p) <= 40:
                    skills.append(p)
    tokens = re.findall(r"[A-Za-z\+\#\-\.\d]{2,}", text)
    ctr = Counter(t.lower() for t in tokens if len(t) > 1)
    common = [k for k, v in ctr.most_common(top_k) if v >= 2]
    skills.extend(common[:top_k])
    seen = set()
    out=[]
    for s in skills:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out

def extract_education(text):
    edu = []
    for kw in COMMON_EDU_KEYWORDS:
        if kw in text.lower():
            edu.append(kw)
    return edu

def extract_projects(text):
    sections = []
    lower = text.lower()
    for kw in PROJECT_SECTIONS:
        if kw in lower:
            sections.append(kw)
    return sections

import numpy as np
def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)))

def _ensure_local_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def get_embedding(texts):
    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True
    if USE_LOCAL_EMB:
        emb = model.encode(texts, convert_to_numpy=True)
        return emb[0] if single else emb
    if openai_client is not None:
        try:
            resp = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
            out = [e.embedding for e in resp.data]
            return out[0] if single else out
        except:
            pass
    try:
        mdl = _ensure_local_model()
        emb = mdl.encode(texts, convert_to_numpy=True)
        return emb[0] if single else emb
    except:
        raise RuntimeError("No embedding provider available. Set OPENAI_API_KEY or USE_LOCAL_EMB.")

def parse_job_description(jd_text):
    jd = {"raw_text": jd_text}
    required=[]
    preferred=[]
    lines = jd_text.splitlines()
    for i, line in enumerate(lines):
        low = line.lower()
        if any(k in low for k in ("requirement","must have","should have","required","skills:","skillset")):
            bits = []
            for j in range(i, min(i+6, len(lines))):
                bits.extend(re.split(r"[,:;•\|\-\/]", lines[j]))
            required.extend([b.strip() for b in bits if b.strip()])
    if not required:
        parts = re.split(r"[,\n]", jd_text)
        required = [p.strip() for p in parts if len(p.strip())<60][:30]
    min_exp = 0
    m = re.search(r"(\d+)\+?\s+years?", jd_text.lower())
    if m:
        min_exp = int(m.group(1))
    jd["required_skills"] = [s for s in required if s]
    jd["preferred_skills"] = preferred
    jd["min_experience"] = min_exp
    jd["title"] = lines[0].strip() if lines else ""
    jd["embed"] = get_embedding(jd_text) if jd_text.strip() else None
    return jd

def parse_resume_file(filepath):
    text = extract_text_from_file(filepath) or ""
    parsed = {}
    parsed["raw_text"] = text
    parsed["filename"] = os.path.basename(filepath)
    parsed["name"] = extract_name(text) or parsed["filename"]
    parsed["contact"] = extract_contact(text)
    parsed["years_experience"] = extract_years_experience(text)
    parsed["skills"] = extract_skills(text)
    parsed["education"] = extract_education(text)
    parsed["projects"] = extract_projects(text)
    parsed["embed"] = get_embedding(text) if text.strip() else None
    return parsed

def normalize_score(x):
    return max(0.0, min(1.0, x))

def score_candidate(parsed_resume, parsed_jd):
    jd_embed = parsed_jd.get("embed")
    res_embed = parsed_resume.get("embed")
    skills_score = 0.0
    if jd_embed is not None and res_embed is not None:
        try:
            skills_score = cosine_sim(jd_embed, res_embed)
        except:
            skills_score = 0.0
    skills_score = normalize_score(skills_score)
    min_exp = parsed_jd.get("min_experience", 0) or 0
    cand_years = parsed_resume.get("years_experience", 0) or 0
    if min_exp == 0:
        experience_score = min(1.0, cand_years / max(1.0, cand_years)) if cand_years>0 else 0.0
    else:
        experience_score = normalize_score(min(cand_years / (min_exp if min_exp>0 else 1.0), 1.0))
    title = parsed_jd.get("title","").lower()
    resume_text = parsed_resume.get("raw_text","").lower()
    title_score = 1.0 if any(tok in resume_text for tok in re.findall(r"\w+", title)) and title else 0.0
    education_score = 1.0 if parsed_resume.get("education") else 0.0
    projects_score = 1.0 if parsed_resume.get("projects") else 0.0
    w = WEIGHTS
    combined = (
        w.get("skills",0)*skills_score +
        w.get("experience",0)*experience_score +
        w.get("title_keywords",0)*title_score +
        w.get("education",0)*education_score +
        w.get("projects",0)*projects_score
    )
    final_score = normalize_score(combined)
    rationale = []
    rationale.append(f"Skills similarity score: {skills_score:.2f}")
    rationale.append(f"Experience score: {experience_score:.2f} (candidate {cand_years} yrs vs required {min_exp} yrs)")
    rationale.append(f"Title/keyword match: {title_score:.2f}")
    rationale.append(f"Education presence: {education_score:.2f}")
    rationale.append(f"Projects presence: {projects_score:.2f}")
    rationale_text = " | ".join(rationale)
    highlights = []
    jd_skills = " ".join(parsed_jd.get("required_skills",[]))[:500]
    for s in parsed_resume.get("skills",[])[:8]:
        if s.lower() in parsed_jd.get("raw_text","").lower():
            highlights.append(f"Matched skill: {s}")
    highlights.append(parsed_resume.get("raw_text","")[:300])
    return {
        "score": final_score,
        "component_scores": {
            "skills": skills_score,
            "experience": experience_score,
            "title_keywords": title_score,
            "education": education_score,
            "projects": projects_score
        },
        "rationale": rationale_text,
        "highlights": highlights,
        "name": parsed_resume.get("name"),
        "filename": parsed_resume.get("filename")
    }

def generate_human_summary(parsed_resume, score_info):
    name = parsed_resume.get("name") or parsed_resume.get("filename")
    score = score_info.get("score", 0.0)
    comp = score_info.get("component_scores", {})
    yrs = parsed_resume.get("years_experience", 0)
    skills_excerpt = ", ".join(parsed_resume.get("skills", [])[:6])
    parts = []
    parts.append(f"{name} — overall fit score {score:.2f}.")
    parts.append(f"Experience: {yrs} years.")
    if comp.get("skills", 0) >= 0.6:
        parts.append("Strong skills match to the JD.")
    elif comp.get("skills", 0) >= 0.35:
        parts.append("Moderate skills match — may require targeted upskilling.")
    else:
        parts.append("Weak skills match.")
    if comp.get("experience",0) >= 0.8:
        parts.append("Meets or exceeds experience requirements.")
    elif comp.get("experience",0) >= 0.4:
        parts.append("Partial experience match.")
    else:
        parts.append("Lacks the required experience.")
    if comp.get("education",0) > 0:
        parts.append("Relevant educational background found.")
    if skills_excerpt:
        parts.append(f"Top extracted skills: {skills_excerpt}.")
    return " ".join(parts)
