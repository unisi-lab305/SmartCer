Create and push a git tag for the current version. No arguments accepted.

Do not enter plan mode. Execute directly.

## Step 1: Read the current version

Run:
```bash
grep '^version = ' pyproject.toml
```

Parse the version as `X.Y.Z`.

## Step 2: Verify the working tree is clean

Run `git status`. If there are uncommitted changes, stop and tell the user:
"There are uncommitted changes. Please commit or stash them before releasing."

## Step 3: Verify version consistency

Run `make version`. If it fails, stop and tell the user the versions are out of sync.

## Step 4: Check the tag does not already exist

Run:
```bash
git tag --list "vX.Y.Z"
```

If the tag already exists, stop and tell the user: "Tag vX.Y.Z already exists. Has the version been bumped?"

## Step 5: Create and push the tag

Run in sequence:
```bash
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
```

Pushing the tag triggers `.github/workflows/release.yml`, which validates the version and the
CHANGELOG entry, runs the tests, builds the packages and creates the GitHub Release with the built
files attached.

## Step 6: Confirm

Report to the user:
- Tag created: `vX.Y.Z`
- Pushed to `origin`
- GitHub Release workflow: `https://github.com/unisi-lab305/SmartCer/actions`
- Install command: `pip install git+https://github.com/unisi-lab305/SmartCer@vX.Y.Z`

**Rules:**
- Do NOT bump the version
- Do NOT modify CHANGELOG.md
- Do NOT create a commit
