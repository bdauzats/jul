# Network

What JuL sends, where, and when. Written for whoever has to open the flows.

A decision never touches the network. The model is read from local disk in a single forward pass,
nothing is generated, and nothing is sent anywhere. The network is used once, to fetch the weights,
and the machine can be cut off after that. There is no JuL server, no account, no API key, and no
telemetry of ours in this repository.

The rest of this page is the detail an audit will ask for.

## 1. Flows to open

Everything is outbound HTTPS on port 443. Nothing inbound, no other protocol, no other port.

| Host | Why | When | Needed at decision time |
| --- | --- | --- | --- |
| `huggingface.co` | model metadata, small files (`config.json`, tokenizer), dataset files | `jul setup`, first load of a model, `jul models add` | no |
| `*.cdn.hf.co` | the weight files themselves (`.safetensors`, served by redirect) | same | no |
| `cas-server.xethub.hf.co` | Xet chunked transfer, the Hub's default for large files | same, unless `HF_HUB_DISABLE_XET=1` | no |
| `pypi.org`, `files.pythonhosted.org` | `jul setup` runs `pip install` for the backend extra | `jul setup` only, skipped with `--no-install` | no |

`github.com` appears in CI and in the "Reproducing the measurements" instructions. The library and
the CLI never contact it.

If your proxy filters by URL rather than by host, the weight request is a 302 from
`huggingface.co/<repo>/resolve/<rev>/<file>` to a signed `*.cdn.hf.co` URL, and the response carries a
`Link: <https://cas-server.xethub.hf.co/...>; rel="xet-reconstruction-info"` header. Both hops have to
be allowed, or the download fails after appearing to start.

## 2. What is sent

Requests are plain GETs for public files, plus the `User-Agent` and `Authorization` headers that
`huggingface_hub` sets. The repositories the presets point at are public, so by default no credential
leaves the machine.

Your states, questions, options, answers and contexts stay in the process. The texts are read from
local disk, turned into vectors in memory, and the vectors stay there. The only bytes that go out are
file requests for public model weights.

`huggingface_hub` ships a telemetry helper, and JuL's code path never reaches it: the download
functions (`snapshot_download`, `hf_hub_download`) do not call it. Set `HF_HUB_DISABLE_TELEMETRY=1`
if you want that written down rather than derived.

Every network call in this repository goes through the Hub client, and `tests/test_network.py` keeps
it that way. No raw HTTP client may be imported under `lib/` or `cli/`, and the code that runs per
decision may not reach the network at all. The test reads the source, so it needs no model, no data
and no network.

## 3. Running with the flows closed

Once the weights are in the cache, JuL works with the network off. Check it on the machine in front
of you:

```bash
jul setup                                    # the one moment the flows are needed
HF_HUB_OFFLINE=1 jul ask choice "Which team should handle this ticket?" \
    -o billing:"payments, invoices" -o technical:"bugs, errors" \
    --state "I was charged twice"
```

`HF_HUB_OFFLINE=1` makes the Hub client fail instead of reaching out, and the decision still returns.
Keep it set in production: an upgrade then cannot quietly start downloading again.

For a machine that must never have the flows opened at all, warm the cache elsewhere and copy it:

```bash
# on a machine that can reach the Hub
HF_HOME=/tmp/jul-cache jul setup --model minicpm5-2b
tar -C /tmp/jul-cache -czf jul-weights.tgz .

# on the isolated machine
mkdir -p /opt/jul-cache && tar -C /opt/jul-cache -xzf jul-weights.tgz
export HF_HOME=/opt/jul-cache HF_HUB_OFFLINE=1
```

The weights are public files and the Hub publishes their SHA-256 digests, so the transfer can be
checked at the destination.

## 4. Pointing at an internal mirror

If the Hub is not allowed but an internal artifact store is, set `HF_ENDPOINT` to it. No code changes
are needed:

```bash
export HF_ENDPOINT=https://hf-mirror.internal.example.com
```

`HF_HUB_DISABLE_XET=1` forces the plain CDN path and removes `cas-server.xethub.hf.co` from the list
above, which helps when the proxy cannot handle the Xet protocol.

## 5. Where files are written

None of this is a network path, but audits ask for it in the same breath.

| Path | Holds |
| --- | --- |
| `~/.cache/huggingface` | downloaded weights: public files, no data of yours (move with `HF_HOME`) |
| `~/.jul/contexts/<name>/` | your data: `examples.txt`, task centers, trained heads |
| `~/.jul/presets/` | presets fitted by `jul models add` |
| `~/.jul/calibration-data/` | the BTZSC calibration sets |

The second row is the one a data-protection review cares about. A saved context holds the example
texts you gave it, word for word, plus vectors computed from them. It is written to local disk, and
nothing in this repository uploads it.
