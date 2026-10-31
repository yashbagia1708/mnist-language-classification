"""Run the whole script and check the headline results are in the expected
range (exact counts can move slightly between library versions and CPUs).

    python tests/check_results.py
"""
import re
import subprocess
import sys

out = subprocess.run([sys.executable, "classification.py"], capture_output=True, text=True, check=True).stdout
full, text, pca = (int(n) for n in re.findall(r"Number of Misclassified Samples: (\d+)", out))
components = int(re.search(r"Principal Components Preserved: (\d+)", out).group(1))

assert 100 <= full <= 180, f"full-pixel errors {full}"
assert 100 <= pca <= 180, f"PCA errors {pca}"
assert 60 <= components <= 90, f"PCA components {components}"
print(f"ok    784 pixels: {full} errors, {components} PCA components: {pca} errors, language demo: {text}/3 wrong")
