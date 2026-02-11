t.py

import re
import src.ArrayGen as q

def syn(expr: str):

    def normalize(expr_: str) -> str:
        expr_ = expr_.replace('^', '**')
        expr_ = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_)
        return expr_

    def parse_math_format(input_str: str):
        match = re.match(r'(f|plot)\((.*)\)', input_str.strip())
        if not match:
            return []

        content = match.group(2).strip()

        # Factorial using !
        if re.match(r'^\d+!$', content):
            number = content.replace("!", "").strip()
            return [number, '!']

        # Factorial using fact:
        if content.startswith("fact:"):
            number = content.split(":")[1].strip()
            return [number, '!']

        content = content.strip("()")

        keywords = []
        parts = re.split(r'\bof\b|\bif\b|:', content)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if re.search(r'[0-9a-zA-Z^]', part):
                part = normalize(part)

            keywords.append(part)

        return keywords


    parsed = parse_math_format(expr)

    if not parsed:
        print("Invalid syntax")
        return

    command = parsed[0]

    # =========================
    # DIAGRAM CASE
    # =========================
    if command == 'diag':
        if len(parsed) < 2:
            print("No expression provided.")
            return

        x, y = q.expr(parsed[1], vars_='x')
        q.Graphing.plot(x, y, static=True)

    # =========================
    # FACTORIAL CASE
    # =========================
    elif command.isdigit():
        if len(parsed) > 1 and parsed[1] == '!':
            z = int(command)
            print(q.Maths.factorial(z))


if __name__ == '__main__':
    expr = 'f(diag of 3x^2+5x+4)'
    syn(expr)
