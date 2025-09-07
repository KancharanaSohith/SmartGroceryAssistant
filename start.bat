@echo off
echo 🛒 Starting Smart Grocery Reorder Assistant...
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting the application...
echo.
echo 🌐 Open your browser and go to: http://localhost:8000
echo.
echo Press Ctrl+C to stop the application
echo.
python main.py
pause
