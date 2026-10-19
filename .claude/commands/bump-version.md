Bump the project version. The argument is: `$ARGUMENTS`

Do not enter plan mode. Execute directly.

## Step 1: Determine bump type

Check `$ARGUMENTS`:
- If `$ARGUMENTS` is `patch`, `minor`, or `major` — use it directly
- If `$ARGUMENTS` is empty or anything else — ask the user: "Which version bump? patch / minor / major"

Do not proceed until the bump type is confirmed.

## Step 2: Read the current version

Run:
```bash
grep '^version = ' pyproject.toml
```

Parse the current version as `MAJOR.MINOR.PATCH` and compute the new version `X.Y.Z`:
- `patch`: increment PATCH only
- `minor`: increment MINOR, reset PATCH to 0
- `major`: increment MAJOR, reset MINOR and PATCH to 0

## Step 3: Read CHANGELOG.md

Read `CHANGELOG.md` and extract the full content of the `## [Unreleased]` section (all subsections and bullets between `## [Unreleased]` and the next `## [` heading).

If `[Unreleased]` is empty, warn the user: "Warning: [Unreleased] section is empty. Proceeding anyway."

## Step 4: Update the two version files

Apply both edits atomically:

1. **`pyproject.toml`**: change `version = "OLD"` → `version = "X.Y.Z"` (double quotes)
2. **`smartcer/__init__.py`**: change `__version__ = 'OLD'` → `__version__ = 'X.Y.Z'` (single quotes)

## Step 5: Update CHANGELOG.md

Rewrite `CHANGELOG.md` so that:
- The `## [Unreleased]` section is left empty (keep the heading, remove all entries below it)
- A new versioned section is inserted immediately after `## [Unreleased]`, before any existing versioned sections:

```
## [X.Y.Z] - YYYY-MM-DD
```

Where `YYYY-MM-DD` is today's date. Move all entries from the old `[Unreleased]` section into this new section, preserving subsection structure (`### Added`, `### Changed`, etc.).

## Step 6: Verify consistency

Run `make version` to confirm both version files are synchronized. If it fails, re-read both files, fix the mismatch, and run `make version` again.

## Step 7: Stage and commit

Stage exactly these files (no others):
```bash
git add pyproject.toml smartcer/__init__.py CHANGELOG.md
```

Commit with:
```bash
git commit -m "chore: bump version to X.Y.Z"
```

No body, no footer, no Co-Authored-By lines.

## Step 8: Confirm

Run `git log --oneline -3` and report to the user:
- New version: `X.Y.Z`
- Files updated: `pyproject.toml`, `smartcer/__init__.py`, `CHANGELOG.md`
- Commit hash and message
