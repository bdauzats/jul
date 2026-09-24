# Publishing jul

`.github/workflows/release.yml` builds the package on every push to `main` and on every published
GitHub release. Same build both times; only the destination differs.

| Trigger | Version published | Index |
|---|---|---|
| push to `main` | `0.1.0.dev<run number>` | TestPyPI, and only if the repo variable `PUBLISH_TESTPYPI` is `true` |
| release published | `jul.__version__`, which must match the tag | PyPI |
| `workflow_dispatch` | dev version | nothing, it builds and stops |

Nothing uploads until the build passes `twine check --strict`, the sdist has rebuilt the wheel on
its own, and the wheel has installed and imported in a clean environment on Linux 3.10, Linux 3.13
and macOS 3.13.

## The version lives in one place

`lib/jul/__init__.py`:

```python
__version__ = "0.1.0"
```

`pyproject.toml` reads it through `[tool.setuptools.dynamic]`, so bumping the module attribute is
the whole release prep. A non-release run appends `.devN` to that same line before building, because
PyPI and TestPyPI both refuse a version that already exists and every push to `main` needs a version
of its own.

If the release tag and `__version__` disagree, the build stops before anything is uploaded.

## One-time setup, on the index side

Both uploads use Trusted Publishing, so there is no API token in this repo and nothing to rotate. It
has to be declared once per index, by whoever owns the project there.

PyPI, at https://pypi.org/manage/account/publishing/ (the "pending publisher" form works before the
project exists, which is the case for the first release):

```
PyPI project name:  jul
Owner:              usejul
Repository name:    jul
Workflow name:      release.yml
Environment name:   pypi
```

TestPyPI, at https://test.pypi.org/manage/account/publishing/, is the same form with environment
name `testpypi`.

Then, in this repo:

- Settings > Environments: create `pypi` and `testpypi`. Worth adding a required reviewer on
  `pypi`, since a PyPI version number can never be reused: the only undo for a bad upload is burning
  the number.
- Settings > Variables: add `PUBLISH_TESTPYPI` = `true` when you want pushes to `main` to upload.
  Without it, `main` still builds and smoke-tests, it just does not publish. Leave it unset on a
  fork.

## Cutting a release

1. Bump `__version__` in `lib/jul/__init__.py`, commit, merge to `main`.
2. Watch the TestPyPI run if the variable is set, or install the artifact from the run by hand.
3. Create the GitHub release with tag `v<version>` and publish it.
4. The `pypi` job uploads. If you put a reviewer on the environment, it waits for you first.

## What it does not cover

The smoke test installs the base package: `import jul`, the assets, the console script. It never
loads a model, so a backend extra that stops resolving on a new Python would go unnoticed until
somebody runs `pip install jul[torch]`. Both backends are exercised by `ci.yml`, which runs on the
same commits, on Linux, for torch. MLX has no runner and never will on the free tier.
