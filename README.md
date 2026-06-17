# EduSense - AI Based Assessment System

EduSense is a Flask and MySQL based intelligent assessment system designed for automated quiz generation and hybrid student evaluation.

The system allows administrators and teachers to manage classes, courses, students, assessments, and MCQs. Students can attempt published assessments, select MCQ answers, and provide written justifications. The system evaluates both objective answers and written reasoning using AI-based scoring with cosine similarity fallback.

## Key Features

* Role-based login system for Administrator, Teacher, and Student
* Class, course, student, and teacher management
* AI-assisted MCQ generation from:

  * Topic name
  * Pasted notes
  * Uploaded TXT, PDF, DOCX, or PPTX files
  * Specific topic/chapter inside uploaded notes
* Assessment-based quiz structure
* Student quiz attempt system
* Hybrid scoring:

  * 0.4 marks for correct MCQ option
  * 0.6 marks for written justification
* LLM API based justification evaluation
* Cosine similarity fallback if AI service is unavailable
* Result detail pages for students and administrators
* MySQL database integration
* Clean academic FYP structure

## Technology Stack

* Python
* Flask
* MySQL
* HTML
* CSS
* JavaScript
* Scikit-learn
* PyPDF2
* python-docx
* python-pptx
* Requests
* Hugging Face Transformers
* Mistral 7B with LoRA adapter support

## Main Modules

### 1. MCQ Generation Module

The MCQ generation module allows a teacher or administrator to generate MCQs from educational content. The system prepares the input, extracts text from uploaded files if needed, selects relevant content for a target topic, and sends the prepared text to an external AI API.

Generated MCQs are not saved directly. They are first shown to the teacher for review and editing. After confirmation, the system creates an assessment and saves the MCQs under that assessment.

### 2. Student Assessment and Scoring Module

Students can view pending assessments for their published courses. During assessment submission, the system checks the selected MCQ option and evaluates the written justification.

The final score for each question is calculated as:

* Correct option: 0.4 marks
* Justification quality: 0.6 marks

The justification is evaluated using an LLM API. If the LLM service is unavailable, the system automatically falls back to TF-IDF cosine similarity.

## Project Structure

```text
EduSense/
│
├── app.py
├── config.py
├── llm_client.py
├── llm_scorer.py
├── requirements.txt
│
├── edusencedatabase/
│   └── edusence.sql
│
├── static/
│   ├── style.css
│   └── images/
│
└── templates/
    ├── base.html
    ├── login.html
    ├── admin/
    └── student/
```

## Database

The project uses MySQL.

Database name:

```text
edusence
```

The exported database file is available at:

```text
edusencedatabase/edusence.sql
```

To run the project locally, import this SQL file into MySQL or phpMyAdmin.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/khubabraza/EduSense-AI-Assessment-System.git
```

### 2. Move into the project folder

```bash
cd EduSense-AI-Assessment-System
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Import the database

Open phpMyAdmin or MySQL and import:

```text
edusencedatabase/edusence.sql
```

### 5. Configure database connection

Open `config.py` and update the database settings if required:

```python
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",
    "database": "edusence",
    "port": 3306
}
```

### 6. Run the Flask app

```bash
python app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

## Demo Login Accounts

### Administrator

```text
Username: admin
Password: admin123
```

### Teacher

```text
Username: farhad
Password: farhad123
```

### Student

```text
Username: khubab
Password: khubab123
```

## AI Integration Note

The project includes API-based AI integration through `llm_client.py`.

The MCQ generation and justification scoring APIs are expected to run externally, such as through a Colab notebook or deployed service. If the LLM scoring API is unavailable, the system safely falls back to cosine similarity so the assessment flow continues working.

The project also includes `llm_scorer.py`, which demonstrates support for local Mistral 7B + LoRA based justification scoring with 4-bit quantization.

## Academic Purpose

This project was developed as a Final Year Project to demonstrate intelligent assessment, automated MCQ generation, and justification-based grading in an educational environment.

## Authors

* Muhammad Khubab Raza
* Muhammad Talha

## Project Status

Completed for academic FYP demonstration and GitHub portfolio presentation.
