@echo off
REM cleanup_project.bat - Script de nettoyage du projet n8n-geospatial

echo Nettoyage du projet n8n-geospatial...

REM Supprimer les fichiers temporaires Python
for /f "delims=" %%i in ('dir /s /b *.pyc 2^>nul') do del "%%i"
for /f "delims=" %%i in ('dir /s /b __pycache__ 2^>nul') do rmdir /s /q "%%i"
for /f "delims=" %%i in ('dir /s /b *.pyo 2^>nul') do del "%%i"

REM Supprimer les sauvegardes
for /f "delims=" %%i in ('dir /s /b *~ 2^>nul') do del "%%i"
for /f "delims=" %%i in ('dir /s /b *.bak 2^>nul') do del "%%i"
for /f "delims=" %%i in ('dir /s /b *.tmp 2^>nul') do del "%%i"

REM Supprimer les fichiers temporaires de l'éditeur
for /f "delims=" %%i in ('dir /s /b .DS_Store 2^>nul') do del "%%i"
for /f "delims=" %%i in ('dir /s /b Thumbs.db 2^>nul') do del "%%i"

echo Nettoyage termine!