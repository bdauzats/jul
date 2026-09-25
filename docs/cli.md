# Command line

```bash
jul ask choice "Which team should handle this ticket?" \
    -o billing:"payments, invoices" -o technical:"bugs, errors" \
    --state "I was charged twice" --model minicpm5-2b

jul run questions.yaml --input tickets.jsonl --output answers.jsonl --context tickets
jul context create tickets --description "Support tickets of an online bank" --examples sample.txt
jul context list                  # the saved contexts
jul context show tickets          # its description, example count, tuned and calibrated questions
jul context delete tickets
jul synth questions.yaml --seeds sample.jsonl --per-option 30 --output synth.jsonl
jul autotune tickets --questions questions.yaml --labeled labeled.jsonl --features hybrid
jul pack bundle/ --questions questions.yaml --context tickets --backend onnx
jul models
jul models add minicpm5-2b-decision --repo usejul/minicpm5-2b-decision-mlx-4bit   # a decision model
jul models add my-model --repo org/Some-Instruct-3B                                 # fits a preset
jul setup --model minicpm5-2b-decision    # backend, weights and one timed decision
```

## File formats

**Questions** (`questions.yaml`, for `run`, `synth`, `autotune`, `pack`): YAML or JSON, one entry per
question, its name as key. `criteria` is `{key: description}` for a `choice`, the ordered levels for a
`score`, nothing for a `noul`.

```yaml
team:
  type: choice
  instructions: Which team should handle this ticket?
  criteria:
    billing: payments, invoices, refunds
    technical: bugs, errors, crashes
is_bug:
  type: noul
  instructions: Does the message report a software bug?
frustration:
  type: score
  instructions: How frustrated is the customer?
  criteria: [Calm, Frustrated but civil, Very angry]
```

**Labeled examples** (`labeled.jsonl`, for `autotune`): one JSON object per line, the text in `state`
(or `text`) and one answer per question in `answers` — the option key for a `choice`, `true`/`false` for
a `noul`, the index of the level for a `score` (0 = the first, lowest level). The answers may also sit at the top level, named after the questions.

```json
{"state": "I was charged twice for my subscription", "answers": {"team": "billing", "is_bug": false, "frustration": 1}}
{"state": "The app crashes every time I export", "answers": {"team": "technical", "is_bug": true, "frustration": 2}}
{"text": "Do you have a yearly plan?", "team": "billing", "is_bug": false, "frustration": 0}
```

**Seeds** (`sample.jsonl`, for `synth --seeds`): the same lines, with `answers` optional — a seed without
answers only informs the style of the texts written.

```json
{"state": "hi, my card got declined at the station again", "answers": {"team": "billing"}}
{"state": "ur app logged me out 3 times today"}
```

**Examples of a context** (`sample.txt`, for `context create --examples`): one text per line, or a
`.jsonl` with a `text` field per line.

```text
I was charged twice for my subscription
The app crashes every time I export
Do you have a yearly plan?
```

**Input of `run`** (`tickets.jsonl`): one message per line, a string or `{"state": ..., "id": ...}`; the
`id` is copied to the output. **Output** (`answers.jsonl`): one line per message, the shape of a Jev API
response:

```json
{"request_id": "c3b355de-…", "model": "e5-small", "usage": {"input_tokens": 9, "output_tokens": 0, "total_tokens": 9}, "answers": {"team": {"choice": "billing", "probabilities": {"billing": 0.91, "technical": 0.06, "sales": 0.03}, "confidence": 0.91}}, "id": "T-1042"}
```
