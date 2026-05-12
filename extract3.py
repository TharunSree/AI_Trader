import re

with open('templates/base.html', 'r', encoding='utf-8') as f:
    base_text = f.read()

# Replace the old javascript we added for jarvis-booting fade
base_split = base_text.split('<script>\n    if (localStorage.getItem(\\'jarvis-dashboard-boot-pending\\') === \\'true\\') {')
if len(base_split) > 1:
    before_script = base_split[0]
    # find where the script block ends
    after_script_start = base_split[1].find('</script>') + 9
    after_script = base_split[1][after_script_start:]
    
    # insert Include before </body>
    new_base = before_script + '{% include \"partials/jarvis_loader.html\" %}' + after_script
    
    with open('templates/base.html', 'w', encoding='utf-8') as f:
        f.write(new_base)
        print("Updated base.html")
