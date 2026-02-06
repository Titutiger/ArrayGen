from pyfiglet import Figlet as q
import sys
import random

ar = sys.argv
q = q()
if len(ar) != 1 and len(ar) != 3:
    sys.exit("Invalid usage")

if len(ar) == 3:
    op = ar[1]
    f_n = ar[2]

    if op not in ('-f', '--font') or f_n not in q.getFonts():
        sys.exit("Invalid usage")

    q.setFont(font=f_n)
else:
    r_f = random.choice(q.getFonts())
    q.setFont(font=r_f)


a = str(input('Input: ')).strip()
print(q.renderText(a))
