import requests
import csv
import time
import json

# API Configuration
API_KEY = "pub_853932f1b84d2d4d5403dcdaf5a91912a7de1"
BASE_URL = "https://newsdata.io/api/1/news"  # Fixed the base URL

# List of countries and categories to iterate
countries = ['in']
categories = ['business', 'entertainment', 'environment', 'food', 'health',
              'politics', 'science', 'sports', 'technology', 'top', 'world']

# CSV setup
csv_file = "Final_one_in.csv"
fieldnames = ["title", "description", "category", "country", "published_at", "link"]

def fetch_news(params):
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        print(f"Response content: {response.text[:200]}")  # Print first 200 chars of response
        return None

with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    
    # Loop over country and category combinations
    for country in countries:
        for category in categories:
            page = 1
            while True:
                params = {
                    "apikey": API_KEY,
                    "country": country,
                    "category": category,
                    "language": "en",
                }
                
                data = fetch_news(params)
                
                if not data or "results" not in data:
                    print(f"Skipping {country}-{category} due to error")
                    break
                
                if not data["results"]:
                    print(f"No results for {country}-{category}")
                    break
                
                for article in data["results"]:
                    writer.writerow({
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "category": category,
                        "country": country,
                        "published_at": article.get("pubDate", ""),
                        "link": article.get("link", "")
                    })
                
                if data.get("nextPage"):
                    page = data["nextPage"]
                    time.sleep(1)  # polite delay for API limits
                else:
                    break

print(f"News data saved to {csv_file}")