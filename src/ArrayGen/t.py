#t.py

import re
from typing import Optional, Any

from maths import *
from graphing import *
from utils import expr, normalize

def syn(expr: str):



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

        content = match.group(2).strip()
        keywords = []

        parts = re.split(r'\bof\b|\bif\b|:', content)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # unwrap only tuple-like variable groups, e.g. "(x,y)" -> "x,y"
            if re.fullmatch(r'\(\s*[A-Za-z]\w*(\s*,\s*[A-Za-z]\w*)+\s*\)', part):
                part = part[1:-1].strip()

            if re.search(r'[0-9a-zA-Z^]', part):
                part = normalize(part)

            keywords.append(part)


        return keywords

    def addressor(parsed_expr: list[str | Any], cmd: str):
        # =========================
        # DIAGRAM CASE
        # =========================
        if cmd == 'diag':
            if len(parsed_expr) < 2:
                print("No expression provided.")
                return

            x, y = expr(parsed_expr[1], vars_='x')
            Graphing.plot(x, y, static=True)

        # =========================
        # FACTORIAL CASE
        # =========================
        elif cmd.isdigit():
            if len(parsed_expr) > 1 and parsed_expr[1] == '!':
                z = int(cmd)
                print(Maths.factorial(z))

    parsed = parse_math_format(expr)
    if not parsed:
        print("Invalid syntax")
        return
    command = parsed[0]

    addressor(parsed, command)


if __name__ == '__main__':
    expr = 'f((x,y) of 3x^2+5x+4)'
    syn(expr)
