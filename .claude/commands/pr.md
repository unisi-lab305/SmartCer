Open a pull request from the current branch into main. The optional argument is: `$ARGUMENTS`

## Step 1: Verify branch

Run `git branch --show-current` to confirm you are NOT on `main`. If you are on `main`, stop and tell the user.

## Step 2: Push branch

Run `git push -u origin HEAD` to ensure the branch is pushed. If it was already up-to-date, continue.

## Step 3: Gather context

Run these commands in parallel:

1. `git log main...HEAD --oneline` — list commits on this branch vs main
2. `git diff main...HEAD --stat` — summarise changed files

## Step 4: Draft title and body

**Title**:

- If `$ARGUMENTS` is non-empty, use it verbatim as the PR title.
- If `$ARGUMENTS` is empty, derive a concise title (under 70 chars) from the commit list.

**Body** (always auto-generated):

```
## Summary
- <bullet per logical change, derived from commits and diff stat>
```

## Step 5: Create the PR

```bash
gh pr create --base main --title "<title>" --body "$(cat <<'EOF'
## Summary
- ...

EOF
)"
```

## Step 6: Return the URL

Print the PR URL returned by `gh pr create`.

---

**Rules:**

- Never add Co-Authored-By lines or AI tool references
- Execute directly, no plan mode
- Do not add `--draft` or other flags unless explicitly requested
- Do not include `Test Plan`
