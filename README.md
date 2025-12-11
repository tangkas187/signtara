# SignSpeak (Demo BISINDO)

Proyek demo untuk menerjemahkan bahasa isyarat BISINDO sederhana ke teks menggunakan:
- **Frontend**: HTML + JS + MediaPipe Hands (CDN).
- **Backend**: Flask API + scikit-learn (KNN classifier).

## Struktur
signspeak_project/
│── README.md  
│  
├── backend/  
│   ├── app.py  
│   ├── model_utils.py  
│   ├── train.py  
│   └── requirements.txt  
│  
└── frontend/  
    ├── index.html  
    ├── style.css  
    └── app.js  

## Cara jalanin
### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate   # (Linux/macOS)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
python app.py
