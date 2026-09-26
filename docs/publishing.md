# Publishing JuL

`.github/workflows/release.yml` builds the package on every push to `main` and on every `v*` tag. The
version comes from git ([setuptools-scm](https://setuptools-scm.readthedocs.io/)), never from a file
edited by hand, and **a release version only comes from a tag**. Git is trunk-based: `main` moves
continuously and publishes an unstable build; a tag cuts a release.

| Trigger | Version built | Index |
|---|---|---|
| push to `main` | the next patch as a dev release: `0.2.1.dev7` = 7 commits after `v0.2.0` | TestPyPI, when the repo variable `PUBLISH_TESTPYPI` is `true` (the unstable channel) |
| tag `v0.3.0rc1` (any pre-release: `aN`, `bN`, `rcN`, `.devN`) | `0.3.0rc1` | TestPyPI |
| tag `v0.3.0` | `0.3.0` | PyPI |
| `workflow_dispatch` | as for `main` | nothing, it builds and stops |

Nothing uploads until the build passes `twine check --strict`, the sdist has rebuilt the wheel on
its own, and the wheel has installed and imported in a clean environment on Linux 3.10, Linux 3.13
and macOS 3.13.

## Where the version comes from

`pyproject.toml` has `[tool.setuptools_scm]`: the build reads the last `v*` tag and the commits since.
On a tagged commit the version is the tag; after it, the next patch with `.dev<commits>`, which pip
sorts after the last release and installs only with `--pre`. The local part (`+g<sha>`) is dropped:
the indexes refuse it. The version is written to `lib/jul/_version.py` at build or install (ignored by
git) and read by `jul.__version__`; a checkout used without installing it says `0+unknown`.

On a tag the workflow checks that the build is exactly the tag: a tag pushed on another commit than
the one built, or a dirty tree, would give a `.devN`, and the build stops before anything is uploaded.

Pushes to `main` are debounced: the TestPyPI job first waits `TESTPYPI_DEBOUNCE_MINUTES` (a repo
variable, 15 by default), and a new push to `main` cancels the run still waiting. A burst of pushes
therefore publishes one dev build, the last one. Tags do not wait and are never cancelled.

To install the unstable channel:

```bash
pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple jul
```

(`--extra-index-url`: JuL's dependencies are not all on TestPyPI.) TestPyPI is not a production
index and may be wiped: it is for trying things, never for depending on.

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
- Settings > Variables: add `PUBLISH_TESTPYPI` = `true` for pushes to `main` to publish the
  unstable build to TestPyPI. Without it, `main` still builds and smoke-tests. Leave it unset on a
  fork. A merge to `main` never reaches PyPI.

## Cutting a release

1. `main` is green (and, if you like, its dev build installs from TestPyPI).
2. Optionally a release candidate first: `git tag v0.3.0rc1 && git push origin v0.3.0rc1`, then
   `pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple jul==0.3.0rc1`.
3. Tag the release commit on `main` and push the tag: `git tag v0.3.0 && git push origin v0.3.0`.
4. The `pypi` job uploads. If you put a reviewer on the environment, it waits for you first. Release
   notes can be written as a GitHub release on the tag afterwards; it triggers nothing.

No file changes for a release: there is no version to bump.

## What it does not cover

The smoke test installs the base package: `import jul`, the assets, the console script. It never
loads a model, so a backend extra that stops resolving on a new Python would go unnoticed until
somebody runs `pip install jul[torch]`. Both backends are exercised by `ci.yml`, which runs on the
same commits, on Linux, for torch. MLX has no runner and never will on the free tier.
