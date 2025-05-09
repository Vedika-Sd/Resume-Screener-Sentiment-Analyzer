import streamlit as st
import nltk
import re
import os
import json
import spacy
import docx
import pdfplumber
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv

# Load env variables
load_dotenv()
API_KEY = "GEMINI_API_KEY"

# Downloads
nltk.download('punkt')
nltk.download('vader_lexicon')

# Configure Gemini API
if not API_KEY:
    st.error("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
    st.stop()
genai.configure(api_key=API_KEY)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("Error loading spaCy model. Try running: python -m spacy download en_core_web_sm")
    st.stop()

# Sentence Transformer model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# ----- Employee Sentiment Analysis -----
class EmployeeSentimentGemini:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def analyze_feedback(self, feedback_list):
        prompt = f"""
You are a smart HR assistant. Given employee feedback comments, extract:

- Sentiment (Positive, Neutral, Negative)
- Key Issues (top 2–3)
- Attrition Risk (Low, Medium, High)
- Recommendations (HR actions)

Return a **valid JSON array only**, without any explanations or markdown, like:
[
  {{
    "feedback": "...",
    "sentiment": "...",
    "key_issues": ["..."],
    "attrition_risk": "...",
    "recommendations": "..."
  }}
]

Feedback List:
{feedback_list}
"""
        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()
                
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
    
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON from Gemini", "raw": response.text}
        except Exception as e:
            return {"error": "Gemini API error", "details": str(e)}
        
# ----- PDF/DOCX Resume Text Extraction -----
def extract_text(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                return "\n".join([page.extract_text() or '' for page in pdf.pages])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"File extraction error: {str(e)}")
        return ""
    return ""

# ----- NLP Preprocessing & Similarity -----
def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def extract_skills_using_spacy(text):
    return [ent.text for ent in nlp(text).ents if ent.label_ in ["ORG", "GPE", "PRODUCT"]]

def compute_similarity(job_desc, resume):
    emb1 = sbert_model.encode(job_desc, convert_to_tensor=True)
    emb2 = sbert_model.encode(resume, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2)[0][0].item()

def generate_prompt(job_desc, resume_text, resume_skills, job_skills, sim_score):
    return f"""
You are an AI-powered HR assistant developed to support human resources teams in screening resumes for the **Software Engineer** role.

Your responsibility is to analyze the following Job Description and Candidate Resume, extract relevant information, and assess the candidate’s suitability.

---

Job Description:
{job_desc}

---

Candidate Resume:
{resume_text}

---

Instructions:

Extract:
- Key Technical Skills
- Total Years of Professional Experience
- Educational Degrees and Qualifications
- Certifications or Specialized Training
- Notable Projects or Achievements

Then evaluate the match:
- Give a Match Score out of 100
- Explain your Reasoning
- List any Missing Skills or Gaps
- Suggest improvements

Semantic Similarity Score (resume vs job): {sim_score:.2f}

---

Respond ONLY with a valid **raw JSON object** like:
{{
  "Skills": [],
  "Experience (years)": "",
  "Degrees": [],
  "Certifications": [],
  "Projects": [],
  "Match Score": "",
  "Reasoning": "",
  "Missing Skills": [],
  "Suggestions": ""
}}

Do NOT include any explanation or markdown formatting like ```json.
"""
def screen_resume_with_prompt(job_desc, resume_text):
    sim_score = compute_similarity(job_desc, resume_text)
    prompt = generate_prompt(job_desc, resume_text, [], [], sim_score)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Remove markdown ```json wrapper if present
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]

        # Now parse JSON safely
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Gemini response is not valid JSON", "raw_response": response.text}
    except Exception as e:
        return {"error": "Gemini API error", "details": str(e)}


# ---------- Streamlit App ----------
st.set_page_config(page_title="Unstop AI HR Assistant", layout="wide")
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Home", "Resume Screening", "Employee Sentiment Analysis"])

if section == "Home":
    st.title("💼 AI-Powered HR Assistant")
    st.markdown("""
Welcome to the **AI HR Assistant**.

- 📑 **Resume Screening**: NLP + Gemini for candidate evaluation
- 📝 **Employee Sentiment Analysis**: Understand employee concerns, attrition risk, and HR action steps
""")

elif section == "Resume Screening":
    st.title("📑 Resume Screening")

    uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx"])
    job_description = st.text_area("Job Description")

    if uploaded_resume and job_description:
        resume_text = extract_text(uploaded_resume)
        st.subheader("Resume Text Preview")
        st.text_area("Resume", resume_text, height=250)

        if st.button("Screen Resume"):
            with st.spinner("Analyzing..."):
                result = screen_resume_with_prompt(job_description, resume_text)
            st.success("✅ Analysis Complete")
            st.json(result)

elif section == "Employee Sentiment Analysis":
    st.title("📝 Employee Sentiment Analysis")

    feedback_input = st.text_area("Paste employee feedback (one per line)")

    if st.button("Analyze Feedback"):
        feedback_list = [f.strip() for f in feedback_input.split('\n') if f.strip()]
        if feedback_list:
            with st.spinner("Analyzing feedback..."):
                model = EmployeeSentimentGemini()
                result = model.analyze_feedback(feedback_list)
            st.success("✅ Analysis Complete")
            st.json(result)
        else:
            st.warning("Please enter some feedback.")
