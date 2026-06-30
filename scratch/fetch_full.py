import urllib.request

url = "https://steamcommunity.com/id/TharunSree/games/?xml=1"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as response:
        html_data = response.read().decode('utf-8', errors='ignore')
    
    with open("scratch/steam_response.html", "w", encoding="utf-8") as f:
        f.write(html_data)
    print("Fetched and saved to scratch/steam_response.html. Length:", len(html_data))
    
    # Check for keywords
    for word in ["private", "error", "login", "sign in", "redirect", "xml", "gamesList"]:
        count = html_data.lower().count(word)
        print(f"Keyword '{word}': {count} occurrences")
except Exception as e:
    print("Error:", e)
