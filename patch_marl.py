import re

with open("QuantumRepeaterMARL.py", "r") as f:
    content = f.read()

# 1. Change Action Space
content = re.sub(
    r'spaces\.MultiBinary\(self\.n - 1 \+ 3 \* self\.num_pairs\)',
    r'spaces.Box(low=0.0, high=1.0, shape=(self.n - 1 + 3 * self.num_pairs,), dtype=np.float32)',
    content
)

# 2. Add threshold inside step()
content = re.sub(
    r'action = np\.array\(action\)\n\s+idx = 0',
    r'action = (np.array(action) > 0.5).astype(int)\n            idx = 0',
    content
)

with open("QuantumRepeaterMARL.py", "w") as f:
    f.write(content)
