# Pushing this artifact to GitHub

This is a FRESH extract (its own git history — never a fork of any private repo). To publish it to the
empty `free-locality` repo:

```bash
cd free-locality
git init -b main
git add .
git commit -m "Initial public artifact release for the Free Locality paper"
git remote add origin https://github.com/astrobillyvazquez/free-locality.git
git push -u origin main
```

After pushing, the CI badge (`.github/workflows/ci.yml`) runs `make figures && make cis && pytest` on
GitHub's runners — a green check is the "claims are reproducible" signal. Paste the repo URL back so it
can be dropped into the paper's reproducibility `\todo{repo URL}`.
