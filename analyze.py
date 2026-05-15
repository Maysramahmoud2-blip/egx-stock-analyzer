import sys
sys.path.insert(0, '/Users/maysre/AI-Learning/StockProject')
from stock_analyzer import StockAnalyzer

symbol = sys.argv[1] if len(sys.argv) > 1 else input("رمز السهم: ")
a = StockAnalyzer(symbol.strip())
a.run()
