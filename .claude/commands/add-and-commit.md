Stage all changes and create a git commit. The argument is: `$ARGUMENTS`

## Step 1: Understand the current state

Run these commands in parallel to gather context:

1. `git status` — identify all modified, added, and deleted files
2. `git diff HEAD` — read the full diff of all changes
3. `git log --oneline -5` — read recent commit messages to match the project's commit style
4. Read `CHANGELOG.md` — inspect the current `## [Unreleased]` section

## Step 2: Update CHANGELOG.md

Before staging anything, update `CHANGELOG.md` to reflect the changes in the diff.

**Rules:**
- Add entries ONLY under the `## [Unreleased]` section
- Use Keep a Changelog subsections: `### Added`, `### Changed`, `### Fixed`, `### Removed`, `### Deprecated`, `### Security` — include only those that apply
- Each entry is a single bullet: `- <concise description of what changed>`
- Do NOT duplicate entries already present in `[Unreleased]`
- Keep bullets concise (one line each), factual, user-facing in tone
- Do NOT mention file names or internal implementation details unless essential
- Write in English

## Step 3: Stage the changes

Stage files individually by name (do NOT use `git add -A` or `git add .`):

- List every file reported by `git status` that should be committed
- Always include `CHANGELOG.md` (modified in Step 2)
- Skip files that likely contain secrets (`.env`, credentials, private keys)
- Run `git add <file1> <file2> ...` with the specific paths

## Step 4: Draft the commit message

Read the diff and recent log carefully, then write a commit message following the rules below.

**CRITICAL rules — never violate these:**
- Do NOT include `Co-Authored-By`, `Generated with`, or any reference to Claude, Anthropic, or AI tools
- Do NOT include any footer lines at all
- Use the conventional commit format: `<type>(<scope>): <description>` where scope is optional

**Mode: check `$ARGUMENTS`**

- If `$ARGUMENTS` is empty or not `verbose`:
  - Write a **single line only** — title only, no blank line, no body, no bullets
  - Keep it under 72 characters
  - Example: `feat(models): add transformer encoder block`

- If `$ARGUMENTS` is `verbose`:
  - Write a **title line** (under 72 chars), then a **blank line**, then a **bullet-point body**
  - The body should explain _what_ changed and _why_, one bullet per logical change
  - Example:
    ```
    feat(models): add transformer encoder block

    - implement multi-head self-attention with configurable heads and dropout
    - add positional encoding with learned embeddings
    - register encoder in MODEL_REGISTRY under key 'transformer_encoder'
    - add type hints and Google-style docstrings to all public methods
    ```

## Step 5: Commit

Run the commit using a HEREDOC to preserve formatting:

```bash
git commit -m "$(cat <<'EOF'
<your drafted message here>
EOF
)"
```

## Step 6: Verify

Run `git log --oneline -3` and confirm the new commit appears at the top with the correct message.
