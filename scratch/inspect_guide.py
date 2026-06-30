import urllib.request
import re

url = "https://gamestegy.com/post/cyberpunk-2077/1038/berserk-build"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as response:
        html_content = response.read().decode('utf-8', errors='ignore')
    
    print("Page length:", len(html_content))
    
    # Let's search for potential main containers, like articles, sections, or classes like 'post', 'guide', 'content', 'main'
    articles = re.findall(r'<article.*?>', html_content, re.IGNORECASE)
    print("Articles found:", len(articles), articles)
    
    # Find all divs with specific classes or IDs
    # Let's look for common article/post classes:
    matches = re.findall(r'<div\s+[^>]*class=["\']([^"\']*(?:content|post|article|body|main|guide|entry)[^"\']*)["\']', html_content, re.IGNORECASE)
    print("Potential container classes:", set(matches[:25]))

    # Let's write the first 5000 chars of the body to inspect
    body_start = html_content.find('<body')
    if body_start != -1:
        with open("scratch/guide_body.html", "w", encoding="utf-8") as f:
            f.write(html_content[body_start:body_start+40000])
        print("Wrote body excerpt to scratch/guide_body.html")
        
except Exception as e:
    print("Error:", e)
