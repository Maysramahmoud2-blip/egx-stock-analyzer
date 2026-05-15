#!/bin/bash
export PATH="/Users/maysre/homebrew/bin:/usr/local/bin:$PATH"
cd /Users/maysre/AI-Learning/StockProject
echo "======= $(date) =======" >> /Users/maysre/AI-Learning/StockProject/daily_log.txt
python3 daily_report.py >> /Users/maysre/AI-Learning/StockProject/daily_log.txt 2>&1
python3 deep_value.py >> /Users/maysre/AI-Learning/StockProject/deep_value_log.txt 2>&1
echo "======= تم =======" >> /Users/maysre/AI-Learning/StockProject/daily_log.txt
