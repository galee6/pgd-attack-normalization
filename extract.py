"""Extracts attack outputs"""

f = open("attack_outputs.txt", "r")
out = open("attack_outputs.csv", "w")
lines = f.readlines()

for l in lines:
    if l.startswith("Attack Success Rate"):
        dash=l.index("-")
        steps=int(l[dash-2:dash].replace("(", ""))
        eqs=l.index("=")
        alpha=float(l[eqs+1:eqs+5])
        asr=float(l[-5:])
        out.write(f"{steps},{alpha},{asr}\n")
    else:
        continue

out.close()