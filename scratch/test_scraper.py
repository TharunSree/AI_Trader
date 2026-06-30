import urllib.request
import re
from html.parser import HTMLParser

class SimpleHTMLScraper(HTMLParser):
    def __init__(self, target_tag='body'):
        super().__init__()
        self.text_list = []
        self.target_tag = target_tag
        self.in_target = False
        self.ignore_tags = {'script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript', 'aside'}
        self.current_tag = None
        self.depth_ignore = 0
        self.target_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == self.target_tag:
            if not self.in_target:
                self.in_target = True
                self.target_depth = 1
            else:
                self.target_depth += 1
        elif self.in_target:
            self.target_depth += 1

        if tag in self.ignore_tags:
            self.depth_ignore += 1
        self.current_tag = tag

    def handle_endtag(self, tag):
        if tag in self.ignore_tags:
            self.depth_ignore = max(0, self.depth_ignore - 1)

        if self.in_target:
            self.target_depth -= 1
            if self.target_depth <= 0:
                self.in_target = False

    def handle_data(self, data):
        if self.in_target and self.depth_ignore == 0:
            text = data.strip()
            if text:
                if self.current_tag in {'h1', 'h2', 'h3', 'h4'}:
                    self.text_list.append(f"\n### {text}\n")
                elif self.current_tag == 'p':
                    self.text_list.append(f"\n{text}\n")
                elif self.current_tag == 'li':
                    self.text_list.append(f"- {text}")
                else:
                    self.text_list.append(text)

url = "https://gamestegy.com/post/cyberpunk-2077/1038/berserk-build"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as response:
        html_content = response.read().decode('utf-8', errors='ignore')
    
    # Detect target tag
    target = 'body'
    if '<article' in html_content.lower():
        target = 'article'
    elif '<main' in html_content.lower():
        target = 'main'
        
    print("Detected target tag:", target)
    
    parser = SimpleHTMLScraper(target_tag=target)
    parser.feed(html_content)
    
    content = "\n".join(parser.text_list)
    content = re.sub(r'\n\s*\n', '\n\n', content)
    
    print("\n=== EXTRACTED PREVIEW (First 1500 chars) ===")
    print(content[:1500])
    print("=============================================")
    print("Total length:", len(content))
    
    # Check if author info or footer is present
    if "Gamestegy Founder" in content:
        print("WARNING: Author info still found!")
    else:
        print("SUCCESS: Author info excluded!")
        
except Exception as e:
    print("Error:", e)
