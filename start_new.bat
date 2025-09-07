@echo off
echo 🛒 Starting Enhanced Smart Grocery Reorder Assistant v2.0...
echo.
echo ✨ New Features:
echo    - Multi-user accounts with pre-loaded test data
echo    - Dynamic brown-based color system
echo    - Indian brands with smart consumption rates
echo    - No manual consumption rate input needed!
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting the application...
echo.
echo 🌐 Open your browser and go to: http://localhost:8000
echo.
echo 🧪 Test Accounts Available:
echo    - Single Professional (1 person)
echo    - Young Couple (2 people)
echo    - Family of 4 (4 people)
echo    - Large Family (6 people)
echo    - Student Hostel (8 people)
echo.
echo Press Ctrl+C to stop the application
echo.
python main.py
pause
