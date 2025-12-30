// ===============================================
// auth.js - Penghubung Halaman & Keamanan
// ===============================================

const TOKEN_KEY = 'signtara_jwt_token';

// Simpan token saat login berhasil
function saveToken(token) {
    localStorage.setItem(TOKEN_KEY, token);
}

// Ambil token untuk verifikasi
function getToken() {
    return localStorage.getItem(TOKEN_KEY);
}

// Logout: Hapus token dan kembali ke login
function logout() {
    localStorage.removeItem(TOKEN_KEY);
    window.location.href = 'login.html';
}

// Cek status login (True/False)
function isLoggedIn() {
    return !!getToken();
}

// Proteksi Halaman: Taruh di paling atas index.html dan permium.html
function requireAuth() {
    if (!isLoggedIn()) {
        window.location.href = 'login.html';
    }
}

// Proteksi Halaman Login: Taruh di login.html agar user tidak login 2x
function redirectIfLoggedIn() {
    if (isLoggedIn()) {
        window.location.href = 'index.html';
    }
}