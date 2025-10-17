# AI-Powered Resume Screener

This project is a sophisticated, AI-powered resume screening application designed to streamline the hiring process. It leverages Natural Language Processing (NLP) and Large Language Models (LLMs) to automatically parse resumes, extract key information, and score candidates against job descriptions.

## Core Features

*   **Resume Parsing:** The application accepts resumes in both `.txt` and `.pdf` formats, automatically extracting the raw text content.
*   **Advanced Skill Extraction:** It uses a powerful skill extraction engine built with spaCy's `PhraseMatcher`. This allows for the identification of a wide range of technical skills, including aliases and variations (e.g., recognizing "k8s" as "Kubernetes"). The skill taxonomy is easily extensible.
*   **AI-Powered Scoring:** The application integrates with Large Language Models (like Google's Gemini or local Flan-T5 models) to provide an intelligent match score (1-10) between a candidate's resume and a given job description. It also provides a brief justification for the score.
*   **Fallback Mechanism:** In case the primary LLM fails, the system gracefully falls back to a cosine similarity score based on text embeddings, ensuring a reliable screening process.
*   **Interactive Frontend:** The user interface is built with Streamlit, providing a simple and intuitive web-based dashboard for uploading resumes, entering job descriptions, and viewing the screening results.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
4.  **Install Spacy small English Model:**
    ```bash
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
    ```


5.  **Configure environment variables:**
    Create a `.env` file in the root of the project and add the following variables:
    ```
    # LLM_MODE can be "GEMINI" or "LOCAL"
    LLM_MODE=LOCAL

    # If using GEMINI, provide your API key
    GEMINI_API_KEY=your-gemini-api-key
    ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app/main.py
    ```

2.  **Open the application in your browser.** The URL will be displayed in your terminal (usually `http://localhost:8501`).

3.  **Upload a resume, enter a job description, and click "Process / Score"** to see the results.

---

##  Video Submission  
 **Project Demo:** [View Video on Google Drive](https://drive.google.com/file/d/1fKuV-saRYMb86S82bDlZjYaP_vCYfAOC/view?usp=sharing)

##  GitHub Repository  
 **Source Code:** [Resume Scanner on GitHub](https://github.com/Simoni03/Resume-_-Scanner)

---

##  Architecture (High-Level)

### **Purpose**
The **Smart Resume Screener** is an AI-powered application that parses resumes, extracts key entities and skills, and evaluates a candidate’s fit for a given job description using a combination of **Large Language Models (LLMs)** and **embedding-based similarity scoring**.

---

###  **Core Components**

#### 1. Streamlit Frontend (`main.py`)
- Handles user interaction for uploading resumes and job descriptions.  
- Displays extracted entities, detected skills, and final LLM-based or embedding-based scores.  

#### 2. Configuration Layer (`config.py`)
- Stores all environment variables and model settings (e.g., embedding model, LLM model, Gemini API key).  
- Ensures consistent access to configuration parameters across modules.  

#### 3. Parsing & NER (`parsers.py`)
- Extracts text from uploaded PDF or TXT resumes.  
- Performs Named Entity Recognition (NER) using **spaCy** and optionally a **BERT-based classifier**.  
- Uses **PhraseMatcher** to identify and categorize skills from a predefined taxonomy.  

#### 4. Embeddings (`embeddings.py`)
- Uses **SentenceTransformers** to generate vector embeddings for resumes and job descriptions.  
- These embeddings are used for **cosine similarity scoring** when LLM-based scoring is unavailable.  

#### 5. LLM Integration (`llm.py`)
- Supports two modes:  
  - **Local Mode:** Uses open-source Hugging Face models (e.g., *Flan-T5*).  
  - **Gemini Mode:** Uses **Google’s Gemini API** for enhanced reasoning and output.  
- Returns structured JSON outputs containing scores and explanations.  

#### 6. Scoring Module (`scoring.py`)
- Uses LLMs to generate a **numeric rating (1–10)** and a **short justification**.  
- Falls back to **cosine similarity–based scoring** if LLM response is invalid or unavailable.  

#### 7. Skill Taxonomy (`skills.py`)
- Maintains a dictionary of standard and alternate skill names.  
- Used by the parser to extract relevant skills via **spaCy’s PhraseMatcher**.  

---

### **Data Flow**

1. User uploads a resume and pastes a job description in the Streamlit UI.  
2. Resume text is extracted and processed for entities and skills.  
3. Embeddings are generated for both the resume and job description.  
4. The LLM or similarity module generates a score (1–10) and justification.  
5. Streamlit displays parsed fields, extracted skills, and the final scores.  

---

###  **Tech Stack**

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **NLP** | spaCy (NER + PhraseMatcher), BERT |
| **Embeddings** | SentenceTransformers |
| **LLMs** | Hugging Face Transformers / Google Gemini API |
| **File Parsing** | pdfminer.six |
| **Language** | Python 3.x |

---
---

##  LLM Prompts Used

This project uses two carefully designed **LLM prompts** to evaluate resumes against job descriptions.  
Each prompt ensures structured JSON output that can be programmatically parsed for reliable scoring and reasoning.

---

###  **Prompt 1 – From `llm.py`**

**Description:**  
Used by the **LLM scoring function** to compare a candidate’s resume with a job description and return a **structured JSON response**.

```text
You are an AI resume screener. Compare this resume to the job description.
Return a JSON object with:
{
  "score": <integer 1-10>,
  "justification": "<brief reason>"
}

Job Title: {job_title}
Job Description: {job_trim}
Resume: {resume_trim}
```
Purpose:
This prompt provides a direct and concise instruction for the LLM to assess candidate-job fit and return a JSON response that can be easily parsed by the system.

###  **Prompt 2 – From `scoring.py`**

**Description:**  
This variant emphasizes strict JSON-only output, ensuring compatibility with programmatic parsing.
You are an expert hiring assistant. Compare the resume below with the job description. 
Return ONLY valid JSON with keys: score (1–10) and justification (short explanation).

```text


Job Title: {job_title}
Job Description: {job_desc}
	
Candidate Resume: {resume_trim}

Return output as strict JSON like:
{"score": 7.5, "justification": "Has Python but lacks AWS experience."}

```
Purpose:
This version enforces strict JSON formatting and includes an example structure.
It ensures concise reasoning and reliable extraction of both numeric score and justification from the LLM output.
