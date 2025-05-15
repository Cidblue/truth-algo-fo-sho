@echo off
setlocal enabledelayedexpansion

echo Testing OCR-based text extraction...

set SOURCE_DIR=E:\Ollama\converted_pdfs
set OUTPUT_DIR=E:\Ollama\extracted_text
set TESSERACT_PATH=C:\Program Files\Tesseract-OCR
set GHOSTSCRIPT_PATH=C:\Program Files\gs\gs9.56.1\bin
set CUSTOM_TEMP_DIR=D:\temp_ocr

REM Create custom temp directory on D: drive
if not exist "%CUSTOM_TEMP_DIR%" mkdir "%CUSTOM_TEMP_DIR%"

REM Set ImageMagick temporary directory to use D: drive
set MAGICK_TEMPORARY_PATH=%CUSTOM_TEMP_DIR%
set MAGICK_TMPDIR=%CUSTOM_TEMP_DIR%

if not exist "%SOURCE_DIR%\*.pdf" (
    echo No PDF files found in %SOURCE_DIR%
    goto :EOF
)

set "PDF_FILE=Admin Dictionary.pdf"
set "TXT_FILE=Admin Dictionary.txt"

if not exist "%SOURCE_DIR%\%PDF_FILE%" (
    echo PDF file not found: %SOURCE_DIR%\%PDF_FILE%
    goto :EOF
)

echo Checking for Tesseract OCR...
where tesseract >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Tesseract not found in PATH. Checking standard installation location...
    if not exist "%TESSERACT_PATH%\tesseract.exe" (
        echo Tesseract OCR not found. Please install it from:
        echo https://github.com/UB-Mannheim/tesseract/wiki
        goto :EOF
    )
    set "TESSERACT_CMD=%TESSERACT_PATH%\tesseract.exe"
) else (
    set "TESSERACT_CMD=tesseract"
)

echo Checking for Ghostscript...
where gswin64c >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Ghostscript not found in PATH. Checking standard installation location...
    if exist "%GHOSTSCRIPT_PATH%\gswin64c.exe" (
        echo Adding Ghostscript to PATH temporarily...
        set "PATH=%PATH%;%GHOSTSCRIPT_PATH%"
    ) else (
        echo Ghostscript not found. Please install it from:
        echo https://ghostscript.com/releases/gsdnld.html
        goto :EOF
    )
)

echo Converting PDF to images for OCR...
echo This may take some time for large documents...

REM Check if ImageMagick is installed
where magick >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ImageMagick not found. Please install it from:
    echo https://imagemagick.org/script/download.php
    goto :EOF
)

REM Create a temporary directory for images on D: drive
set "TEMP_DIR=%CUSTOM_TEMP_DIR%\pdf_ocr_%RANDOM%"
mkdir "%TEMP_DIR%" 2>nul

echo Extracting pages from PDF to images...
REM Lower density and quality to save space
magick -density 150 "%SOURCE_DIR%\%PDF_FILE%" -quality 75 -depth 8 "%TEMP_DIR%\page-%%03d.png"

echo Running OCR on extracted images...
echo This will take some time for large documents...

REM Create empty output file
echo. > "%OUTPUT_DIR%\%TXT_FILE%"

REM Process each image with Tesseract
for %%f in ("%TEMP_DIR%\*.png") do (
    echo Processing %%~nf...
    "%TESSERACT_CMD%" "%%f" "%%f" -l eng
    type "%%f.txt" >> "%OUTPUT_DIR%\%TXT_FILE%"
    del "%%f.txt" 2>nul
    del "%%f" 2>nul
)

echo Cleaning up temporary files...
rmdir /s /q "%TEMP_DIR%" 2>nul

if exist "%OUTPUT_DIR%\%TXT_FILE%" (
    echo Text extraction successful.
    for %%I in ("%OUTPUT_DIR%\%TXT_FILE%") do set TEXT_SIZE=%%~zI
    echo Text file size: !TEXT_SIZE! bytes
    echo First few lines of extracted text:
    type "%OUTPUT_DIR%\%TXT_FILE%" | find /v "" /n | findstr "^\[[1-5]\]"
) else (
    echo Text extraction failed.
)

echo.
echo OCR process complete.
pause
endlocal


