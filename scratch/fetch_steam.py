import urllib.request

url = "https://steamcommunity.com/id/TharunSree/games/?xml=1"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as response:
        xml_data = response.read().decode('utf-8', errors='ignore')
    
    lines = xml_data.split('\n')
    print("Total lines:", len(lines))
    if len(lines) >= 58:
        print("Line 57:", lines[56])
        print("Line 58:", lines[57])
        print("Line 58 length:", len(lines[57]))
        if len(lines[57]) >= 88:
            print("Around column 88:", lines[57][60:110])
        print("Line 59:", lines[58])
    else:
        print("XML too short!")
except Exception as e:
    print("Error:", e)
