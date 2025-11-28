# app/model_config.py

WEIGHTS = {
    "skills": 0.40,
    "experience": 0.30,
    "title_keywords": 0.15,
    "education": 0.10,
    "projects": 0.05
}

# Thresholds & misc
MIN_PASS_SCORE = 0.60    # 0-1 scale; candidate scoring >= this can be shortlisted
MAX_RESUMES_BATCH = 50   # demo default
