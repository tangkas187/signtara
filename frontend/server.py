from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime
import jwt
import json
import os

app = Flask(__name__)
CORS(app)

SECRET_KEY = "rahasia_signtara_123"
DB_FILE = "users.json" # Nama file untuk menyimpan data akun

# --- FUNGSI BANTUAN DATABASE ---

def load_users():
    """Membaca data user dari file JSON"""
    if not os.path.exists(DB_FILE):
        return {} # Jika file belum ada, kembalikan kosong
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_users(users_data):
    """Menulis data user ke file JSON"""
    with open(DB_FILE, 'w') as f:
        json.dump(users_data, f, indent=4)

# -------------------------------

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "storage": "JSON File"})

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')

    if not email or not username or not password:
        return jsonify({"error": "Semua kolom harus diisi!"}), 400

    # 1. BUKA DATABASE
    users_db = load_users()

    # 2. CEK USERNAME
    if username in users_db:
        return jsonify({"error": "Username sudah digunakan!"}), 400

    # 3. TAMBAH USER BARU
    users_db[username] = {
        "email": email,
        "password": password 
    }

    # 4. SIMPAN KEMBALI KE FILE
    save_users(users_db)

    print(f"[INFO] User baru tersimpan di {DB_FILE}: {username}")
    return jsonify({"message": "Registrasi berhasil!"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    # 1. BUKA DATABASE UNTUK CEK
    users_db = load_users()
    user = users_db.get(username)

    # 2. VALIDASI
    if not user or user['password'] != password:
        return jsonify({"error": "Username atau password salah!"}), 401

    # 3. BUAT TOKEN
    token = jwt.encode({
        'user': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, SECRET_KEY, algorithm="HS256")

    print(f"[INFO] User login: {username}")
    return jsonify({
        "message": "Login berhasil",
        "access_token": token
    })

if __name__ == '__main__':
    print(f"Server berjalan. Data akun disimpan di file '{DB_FILE}'")
    app.run(debug=True, port=5000)