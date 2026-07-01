import os

def check_template(filepath):
    print(f"Checking template: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    stack = []
    for idx, line in enumerate(lines):
        line_num = idx + 1
        # Extract template tags
        import re
        tags = re.findall(r'{%\s*(.*?)\s*%}', line)
        for tag in tags:
            parts = tag.split()
            if not parts:
                continue
            cmd = parts[0]
            
            if cmd in ['if', 'for', 'block', 'with', 'comment']:
                stack.append((cmd, line_num, line.strip()))
            elif cmd.startswith('end'):
                expected = cmd[3:]
                if not stack:
                    print(f"Error: Unmatched '{cmd}' on line {line_num}")
                    print(f" -> Line content: {line.strip()}")
                    continue
                top_cmd, top_line, top_content = stack.pop()
                if top_cmd != expected:
                    print(f"Error: Mismatched '{cmd}' on line {line_num} (expected close for '{top_cmd}' from line {top_line})")
                    print(f" -> Opening line {top_line}: {top_content}")
                    print(f" -> Closing line {line_num}: {line.strip()}")
                    # Push back to preserve stack order
                    stack.append((top_cmd, top_line, top_content))
            elif cmd == 'else' or cmd == 'elif':
                # Check if we are inside an 'if'
                if not any(x[0] == 'if' for x in stack):
                    print(f"Error: '{cmd}' outside of 'if' block on line {line_num}")

    if stack:
        print(f"Error: Unclosed tags remaining at end of file:")
        for cmd, line, content in stack:
            print(f" -> Line {line} ({cmd}): {content}")
    else:
        print("Success: All tags are perfectly balanced!")

if __name__ == '__main__':
    template_path = os.path.join('templates', 'relax.html')
    if os.path.exists(template_path):
        check_template(template_path)
    else:
        print("File not found.")
