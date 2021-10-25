terms = 30
out = ""
for i in range(terms):
    if i:
        out += "+"
    out += "f"
    for _ in range(i):
        out += "'"
    out += f"(0)/{i}!*g(x, {i})"

print(out)