import re

with open("main.py", "r") as f:
    code = f.read()

# Remove all triple-quoted docstrings (both """ and ''')
code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)

# Remove single-line comments
code = re.sub(r'^\s*#.*$', '', code, flags=re.MULTILINE)

# Remove extra blank lines (3 or more newlines to 2)
code = re.sub(r'\n{3,}', '\n\n', code)

with open("cleaned_script.py", "w") as f:
    f.write(code)
