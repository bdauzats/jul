# Adapting JuL to your data

Three tools, from cheapest to most effective: a `Context` (unlabeled examples), `autotune(...)`
(labeled examples), and `jul synth` (when there are too few labels).

## Context — what the data looks like

Jev has no equivalent. A context acts at three levels.

```python
from jul import TypeSafeClient, Context

tickets = Context(
    description="Customer support tickets of an online bank, written in English by customers.",
    examples=open("sample_tickets.txt").read().splitlines(),   # ~50–200 real texts, no labels
)

client = TypeSafeClient(context=tickets)                       # for every call
client.system_one(state, questions, context=tickets)           # or per call
```

| Element       | What it does                                     | Cost                            | Status                                                                                                        |
| ------------- | ------------------------------------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `description` | prepended as `Context: …` to both formulations   | nil (sits in the cached prefix) | **off by default, measured harmful on average** — `use_description=True` to try it                            |
| `examples`    | their mean vector becomes the center of the task | computed once                   | measured: helps on topics (AG News 0.66 → 0.75), slightly hurts on fine-grained tasks (Banking77 0.55 → 0.53) |
| `labeled` | fits a temperature and a per-option bias | computed once | proven, but makes any comparison with Jev unfair |

A task center is the best center measured, which is the main
reason to bother with a context at all. **Ten examples already capture most of the gain, fifty is the
sweet spot, two hundred adds nothing**. Below ten the center is noise and can be
worse than no context at all — with 5 examples, a ticket rated `billing` at 0.98 flipped to a wrong
`technical`. `Context` warns under ten.

The gain is uneven: topics gain the most, while financial sentences lost 3 points. A task center helps most when your texts have a style of their own.

**The `description` is off by default.** Measured over four datasets, it failed the plan's bar:
MiniCPM gained 1.3 points on average with one dataset losing 4, and a 9B model lost on all four
(−5.2 on average). Worse, the sign flips between models on the same data. Switch it on with
`use_description=True` only if you measure a gain on your own.

Adding information to a prompt does not always help: listing the options helped on topics and
emotions but hurt on sentiment. Measure before trusting the description.

Contexts are cached on disk under `~/.jul/contexts/<name>/` and can be reused by name:

```python
client.system_one(state, questions, context="tickets")
```

## `autotune(...)` — when it is off-key

Give it labeled examples and it stops missing the note. A small head over the vectors the method
already computes: **the LLM itself is not modified**, training takes seconds, and inference stays
exactly as fast — the correction is applied after the fact, and nobody hears it.

```python
report = client.autotune("tickets", questions, labeled)   # labeled: [(state, {"team": "billing"}), …]
print(report["team"])
client.system_one(state, questions, context="tickets")   # uses the head automatically
```

```
question 'topic': 200 labeled examples, 4 options
  held out for judging : 40
  zero-shot accuracy   : 0.800
  tuned head accuracy  : 0.925
  result               : ACTIVE (beats zero-shot on the held-out examples (+0.125))
```

**Safety net**: the head is judged by stratified cross-validation, so every labeled example is
predicted by a head that never saw it. If it does not beat the zero-shot method there, it is not
activated and the report says so. On six measured curves it decides right 19 times out of 22, and its
three errors are each worth under 3 points.

| Labeled examples per question | What happens                                                               |
| ----------------------------- | -------------------------------------------------------------------------- |
| below `max(20, 3 × options)`  | calibration only (temperature + per-option bias)                           |
| above it                      | a head is trained, and kept only if it beats zero-shot in cross-validation |

The floor scales with the number of options because that is what the measurement showed: 200 examples
is plenty for 4 options and not enough for Banking77's 72. Measured gains at 1000 examples (on a
validation split, ±3.5 points):

|             | AG News       | Emotion       | Banking77     |
| ----------- | ------------- | ------------- | ------------- |
| MiniCPM5-2B | 0.770 → 0.870 | 0.480 → 0.630 | 0.555 → 0.695 |

MiniCPM needs 500 to 1000 examples to plateau. A model with more separable vectors learns from
fewer: on the Jev bench, `wemm-4b-4bit` goes from 0.857 to 0.897 with 1000.

A head only knows the options it saw, and only the preset whose vectors it saw: change either and you
call `autotune(...)` again.

### What the head reads: `features=`

```python
client.autotune("tickets", questions, labeled, features="hybrid")   # "vector" (default), "lexical", "hybrid"
```

- **vector** — the model's vectors, as above.
- **lexical** — TF-IDF of the text (words 1-2 and characters 2-5, `lib/jul/lexical.py`), no model at all
  in the head. A baseline worth having: it is what a head must beat.
- **hybrid** — both, one head over the concatenation, the weight of each part chosen on an inner split.
  What the vectors miss, the words often carry: on dair-ai/emotion, Harrier 0.6B vectors alone gave
  0.734, TF-IDF alone 0.751, the hybrid head 0.797.

`vector` is the default. `hybrid` usually does better: on the Jev bench, `e5-small` goes from 0.670
with a vector head to 0.770 with a hybrid one.

Lexical and hybrid heads train with scikit-learn (`pip install "jul[tune]"`) and run with numpy alone:
the vocabulary and idf are saved with the head. The same safety net applies: a head that does not beat
zero-shot in cross-validation is not activated.

### Which prompts: `formulations=`

A preset reads each question with its formulations (two for the built-in presets). `autotune` may pick
others, by name, for all questions or per question, and the head remembers them: answering, and
`jul pack`, use exactly those.

```python
client.autotune("tickets", questions, labeled, features="hybrid",
                formulations={"mood": ["question_options"], "intent": ["question"]})
```

| Name | Prompt | Passes per state |
| --- | --- | --- |
| `one_word` | the state alone, no question: shared by every question of a call | 1 for all questions |
| `question_options` | the question and its options | 1 per question |
| `question` | the question without its options: a short prefix whatever the number of options | 1 per question |

With a tuned head, listing the options mattered for 6 emotions (0.884 against 0.852) and not for
Banking77's 72 intents (0.900 with, 0.910 without). `one_word` for every question is the fastest (one
pass per message, whatever the number of questions) and gave 0.791 / 0.890 on those two tasks.

### Not enough labeled examples? `jul synth`

Writes a synthetic labeled dataset from a few real examples, in the format `autotune` reads. It only
writes the data: tuning stays a separate step.

```bash
jul synth questions.yaml --seeds sample.jsonl --per-option 30 --output synth.jsonl --writer <mlx-lm repo>
jul autotune tickets --questions questions.yaml --labeled synth.jsonl
```

The writer first describes the source from the seeds (who writes, form, tone), then writes texts option
by option, shown the whole option list and the seeds carrying that option. Seeds are `{state, answers}`
lines; `answers` may be left out, the text then only informs the style. Synthetic texts are cleaner
than real ones: keep real labeled examples aside to check what a head trained on them is worth.
