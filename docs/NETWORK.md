# Network

What JuL sends, where, and when — written for the people who have to open the flows.

**Short version: a decision never touches the network.** The model is loaded from local disk and read
in a single forward pass; nothing is generated and nothing is sent anywhere. The network is used
once, to *fetch* the weights, and after that the machine can be cut off. There is no JuL server, no
account, no API key, and no telemetry of ours anywhere in this repository.

The rest of this page is the detail an audit will ask for.

## 1. Flows to open

Everything is outbound HTTPS on port 443. Nothing inbound. No other protocol, no other port.

| Host | Why | When | Needed at decision time |
| --- | --- | --- | --- |
| `huggingface.co` | model metadata, small files (`config.json`, tokenizer), dataset files | `jul setup`, first load of a model, `jul models add` | no |
| `*.cdn.hf.co` | the weight files themselves (`.safetensors`, served by redirect) | same | no |
| `cas-server.xethub.hf.co` | Xet chunked transfer, the Hub's default for large files | same, unless `HF_HUB_DISABLE_XET=1` | no |
| `pypi.org`, `files.pythonhosted.org` | `jul setup` runs `pip install` for the backend extra | `jul setup` only, skipped with `--no-install` | no |

`github.com` appears only in CI and in the "Reproducing the measurements" instructions. It is never
contacted by the library or by the CLI.

If your proxy filters by URL rather than by host, the weight request is a 302 from
`huggingface.co/<repo>/resolve/<rev>/<file>` to a signed `*.cdn.hf.co` URL, and the response carries a
`Link: <https://cas-server.xethub.hf.co/...>; rel="xet-reconstruction-info"` header. Both hops must be
allowed, or the download fails after appearing to start.

## 2. What is sent

Requests are plain GETs for public files, plus the `User-Agent` and `Authorization` headers that
`huggingface_hub` sets. No token is required for the public repositories the presets point at, so by
default no credential leaves the machine.

**No state, no question, no option, no answer, and no context ever leaves the process.** Your texts
are read from local disk, turned into vectors in memory, and the vectors stay there. The only bytes
that go out are file requests for public model weights.

`huggingface_hub` ships a telemetry helper, and it is not on the code path JuL uses: the download
functions (`snapshot_download`, `hf_hub_download`) do not call it. Set `HF_HUB_DISABLE_TELEMETRY=1`
if you want the guarantee written down rather than derived.

## 3. Every call site, in this repository

There are eleven, all of them in the Hugging Face client, none of them on the decision path:

| File | Line | Call | Reached by |
| --- | --- | --- | --- |
| `cli/jul_cli/setup.py` | 118, 126 | `snapshot_download` | `jul setup` |
| `cli/jul_cli/setup.py` | 134, 142 | `snapshot_download(local_files_only=True)` | cache probe — offline, never downloads |
| `cli/jul_cli/main.py` | 222 | `scan_cache_dir` | `jul models` — reads the local cache only |
| `lib/jul/backbone.py` | 126 | `snapshot_download` | first load of a model |
| `lib/jul/backends/mlx.py` | 28 | `hf_hub_download("config.json")` | first load, MLX |
| `lib/jul/backends/torch.py` | 55, 56 | `from_pretrained` | first load, PyTorch |
| `lib/jul/decision.py` | 100 | `hf_hub_download("decision.json")` | a decision model |
| `lib/jul/presets.py` | 127 | `snapshot_download` | `jul models add` on a decision model |
| `lib/jul/calibrate.py` | 95, 102 | `load_dataset` (BTZSC) | `jul models add`, unless `--data <dir>` |

`tests/test_network.py` asserts this list stays exact: a new network call anywhere under `lib/` or
`cli/` fails the test until it is documented here. It needs no model and no network to run.

## 4. Running with the flows closed

Once the weights are in the cache, JuL works with the network off. Prove it on the machine in front
of you rather than taking this page's word for it:

```bash
jul setup                                    # the one moment the flows are needed
HF_HUB_OFFLINE=1 jul ask choice "Which team should handle this ticket?" \
    -o billing:"payments, invoices" -o technical:"bugs, errors" \
    --state "I was charged twice"
```

`HF_HUB_OFFLINE=1` makes the Hub client fail rather than reach out. The decision still returns, which
is the whole point. Keep it set in production: it turns "we believe it is offline" into a process
that cannot silently start downloading after an upgrade.

For a machine that must never have the flows opened at all, warm the cache elsewhere and copy it:

```bash
# on a machine that can reach the Hub
HF_HOME=/tmp/jul-cache jul setup --model minicpm5-2b
tar -C /tmp/jul-cache -czf jul-weights.tgz .

# on the isolated machine
mkdir -p /opt/jul-cache && tar -C /opt/jul-cache -xzf jul-weights.tgz
export HF_HOME=/opt/jul-cache HF_HUB_OFFLINE=1
```

The weights are public files with published SHA-256 digests on the Hub, so the transfer can be
checked at the destination.

## 5. Pointing at an internal mirror

If the Hub is not allowed at all but an internal artifact store is, set `HF_ENDPOINT` to it and no
code changes are needed:

```bash
export HF_ENDPOINT=https://hf-mirror.internal.example.com
```

`HF_HUB_DISABLE_XET=1` forces the plain CDN path and removes `cas-server.xethub.hf.co` from the list
above — useful when the proxy cannot handle the Xet protocol.

## 6. Where files are written

Nothing here is a network path, but audits ask for it in the same breath.

| Path | Holds |
| --- | --- |
| `~/.cache/huggingface` | downloaded weights — public files, no data of yours (move with `HF_HOME`) |
| `~/.jul/contexts/<name>/` | **your data**: `examples.txt`, task centers, trained heads |
| `~/.jul/presets/` | presets fitted by `jul models add` |
| `~/.jul/calibration-data/` | the BTZSC calibration sets |

The second row is the one that matters for a data-protection review: a saved context contains the
example texts you gave it, verbatim, plus vectors derived from them. It is written to local disk and
is never uploaded by anything in this repository.
