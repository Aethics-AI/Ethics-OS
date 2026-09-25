# Third-party licences

Licences of everything `aethics-eval` depends on at runtime, i.e. what a user
receives from `pip install aethics-eval`.

Generated from a clean core-only install with `pip-licenses`. Development
tooling (`[dev]`) and local inference (`[local]`) are listed separately below
because they are not installed by default and are not redistributed.

Regenerate with:

```bash
pip install -e . && pip install pip-licenses
pip-licenses --format=markdown --with-urls
```

For dataset licences — a separate and more consequential question — see
[docs/datasets.md](docs/datasets.md).

---

## Runtime dependencies

Direct: `datasets`, `httpx`, `numpy`, `pydantic`. The rest are transitive.

| Package | Version | Licence |
|---|---|---|
| aiohappyeyeballs | 2.7.1 | PSF |
| aiohttp | 3.14.3 | Apache-2.0 AND MIT |
| aiosignal | 1.4.0 | Apache-2.0 |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.14.2 | MIT |
| attrs | 26.1.0 | MIT |
| certifi | 2026.7.22 | MPL-2.0 |
| charset-normalizer | 3.5.0 | MIT |
| click | 8.4.2 | BSD-3-Clause |
| datasets | 4.8.5 | Apache-2.0 |
| dill | 0.4.1 | BSD-3-Clause |
| filelock | 3.32.2 | Unlicense / MIT |
| frozenlist | 1.8.0 | Apache-2.0 |
| fsspec | 2026.2.0 | BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| hf-xet | 1.6.0 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| huggingface_hub | 1.27.0 | Apache-2.0 |
| idna | 3.18 | BSD-3-Clause |
| multidict | 6.7.1 | Apache-2.0 |
| multiprocess | 0.70.19 | BSD-3-Clause |
| numpy | 2.5.2 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pandas | 3.0.5 | BSD-3-Clause |
| propcache | 0.5.2 | Apache-2.0 |
| pyarrow | 25.0.1 | Apache-2.0 |
| pydantic | 2.13.4 | MIT |
| pydantic_core | 2.46.4 | MIT |
| python-dateutil | 2.9.0.post0 | Apache-2.0 AND BSD-3-Clause |
| PyYAML | 6.0.3 | MIT |
| requests | 2.34.2 | Apache-2.0 |
| six | 1.17.0 | MIT |
| tqdm | 4.70.0 | MPL-2.0 AND MIT |
| typing-inspection | 0.4.4 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| urllib3 | 2.7.0 | MIT |
| xxhash | 4.0.0 | BSD-2-Clause |
| yarl | 1.24.5 | Apache-2.0 |

### Notes for legal review

Everything above is permissive (MIT / BSD / Apache-2.0 / PSF) except two
weak-copyleft entries, both of which are **file-level** copyleft that attaches
to the dependency's own files and does not propagate to code that merely
imports it:

- **certifi** — MPL-2.0
- **tqdm** — MPL-2.0 AND MIT

Neither is statically linked or modified here; both are ordinary runtime
imports. No dependency carries a strong copyleft licence (GPL/AGPL/LGPL).

Nothing in this set conflicts with a permissive licence on `aethics-eval`
itself.

---

## Optional: `[local]` extra

Not installed by default. Only pulled by users running open-weight models
locally for exact log-probabilities.

| Package | Licence |
|---|---|
| torch | BSD-3-Clause |
| transformers | Apache-2.0 |

Both permissive. These pull a large transitive tree of their own which is not
enumerated here, as it is not redistributed by this package.

---

## Optional: `[dev]` extra

Not shipped to users. Listed for completeness.

| Package | Licence |
|---|---|
| pytest | MIT |
| pytest-asyncio | Apache-2.0 |
| import-linter | BSD-2-Clause |
| grimp | BSD-2-Clause |
| ruff | MIT |
| mypy | MIT |

---

## This package's own licence

Not yet assigned. Pending a decision from the project lead — Apache-2.0 is the
standing recommendation, for the patent grant. Until it is set,
`pyproject.toml` declares `Apache-2.0`.

See POS-2 and §7 of the open-source work order.
