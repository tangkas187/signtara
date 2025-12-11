@echo off
title 🔄 Reset Model SignSpeak
echo.
echo ==========================================
echo     🔥 RESET MODEL & FILE SEMENTARA 🔥
echo ==========================================
echo.

:: Pindah ke direktori backend
cd /d "%~dp0backend"

:: Pastikan folder models & temp_videos ada
if not exist models (
    mkdir models
)
if not exist temp_videos (
    mkdir temp_videos
)

echo 🗑️ Menghapus file model lama...
del /q models\*.h5 2>nul
del /q models\*.json 2>nul
del /q models\*.joblib 2>nul

echo 🧼 Menghapus file video sementara...
del /q temp_videos\*.webm 2>nul
del /q temp_videos\*.mp4 2>nul

echo.
echo ✅ Semua file model dan cache video telah dihapus.
echo 📁 Folder models dan temp_videos sudah bersih.
echo.
pause
