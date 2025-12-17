# NLP Backend – Indonesian Text Correction & NER
Backend NLP berbasis FastAPI untuk koreksi teks Bahasa Indonesia dan Named Entity Recognition (NER) menggunakan model bert-base-indonesian-NER.

## Model
- Author / Source: cahya
- Model: bert-base-indonesian-NER

## Supported File Types
- .txt
- .pdf
- .docx

## Project Structure
nlp-backend/  
├── app/  
│ ├── core/             # Core NLP correction logic (AdvancedCorrector)  
│ ├── utils/            # File parsing utilities (PDF/DOCX parsers)  
│ └── main.py           # FastAPI application and route definitions  
├── data/               # Directory for generated dictionary storage  
├── requirements.txt    # Project dependencies  
└── run.py              # Server entry point script

## Environment Setup

### Create Virtual Environment
python -m venv venv

### Activate Virtual Environment
Windows
venv\Scripts\activate

Linux / MacOS
source venv/bin/activate

### Install Dependencies
pip install -r requirements.txt

## Running the Server
python run.py

Server will be available at:
http://0.0.0.0:8080

Swagger documentation:
http://0.0.0.0:8080/docs

## API Endpoints

### Correct Raw Text
Method: POST  
Path: /correct-raw  
Content-Type: application/json

Request Example:
{
    "text": "laporan dari budi santoso mengenai proyek di papua pegunungan"
}

Description:
Mengirim teks mentah (raw) dalam format JSON untuk dikoreksi oleh sistem NLP.

### Correct File Text
Method: POST  
Path: /correct-file  
Content-Type: multipart/form-data  
Form field: file

Supported file types:

.txt

.pdf

.docx

Description:
Mengunggah file teks, PDF, atau dokumen Word untuk diparsing dan dikoreksi.

### Example cURL (Raw Text)
curl -X POST http://0.0.0.0:8000/correct-raw

-H "Content-Type: application/json"
-d '{"text":"laporan dari budi santoso mengenai proyek di papua pegunungan"}'

### Example cURL (File Upload)
curl -X POST http://0.0.0.0:8000/correct-file

-F "file=@/path/to/file.pdf"

## License
Copyright (c) 2025 Muhammad Rafly Ash Shiddiqi, Arif Athaya Harahap, Ariiq Tsany Zu, Fadhlullah Akmal