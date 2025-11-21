import requests
import json
import sys

def test_analyze_pair(symbol_a, symbol_b, timeframe="5 mins"):
    url = "http://localhost:8000/api/research/analyze-pair"
    payload = {
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "timeframe": timeframe,
        "lookback_days": 5
    }
    
    print(f"Testing {symbol_a}-{symbol_b}...")
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response keys:", data.keys())
            
            if "spread_series" in data:
                spread = data["spread_series"]
                print(f"Spread length: {len(spread)}")
                print(f"First 5 spread: {spread[:5]}")
                
                # Check for NaNs in raw text if json parser handled it
                if "NaN" in response.text:
                    print("WARNING: NaN found in response text!")
            else:
                print("spread_series NOT found in response")
                
            if "zscore_series" in data:
                zscore = data["zscore_series"]
                print(f"Z-Score length: {len(zscore)}")
                print(f"First 5 zscore: {zscore[:5]}")
            else:
                print("zscore_series NOT found in response")
                
        else:
            print("Error:", response.text)
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    # Use symbols that likely exist
    test_analyze_pair("AVB", "EQR")
