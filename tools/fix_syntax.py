import sys
path = sys.argv[1] if len(sys.argv) > 1 else "scripts/run_failure_mode_coverage.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()


fixed = 0
for i, line in enumerate(lines):
    if 'last_rd.get("c_i_norm", 0.0) / max(last_rd.get("gradient_norm", 1e-12), 1e-12)' in line:
        j = i + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j < len(lines) and not lines[j].strip().startswith(")"):
            lines[i] = line.rstrip() + ")" + "\n"
            fixed += 1
            print(f"Fixed line {i+1}")

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)


print(f"Done. Fixed {fixed} lines.")
