# Live mode — design notes

**Status: not built. Scoped, not committed to.**

The idea: switch it on during a live debate and get feedback as it happens —
fallacies named, the weak joint in what the other side just said, and a line
you could actually use in reply.

This records what would have to be true for that to work, so the decision can
be made on the facts rather than on enthusiasm.

## What already transfers

Most of the thinking is done. The prompts that name fallacies, find the
weakest flank, and compose the best reply are written and working: a run on a
2h21m debate produced 61 records, and every one of the 62 records in the
second run carried a `best_reply`. The judgement layer is not the problem.

## What does not transfer

**Latency is a different pipeline, not a tuning problem.** Measured on the
development machine, batch analysis runs at roughly **3 minutes per 12,000
characters** via claude-cli. Live coaching is useless above about **5–10
seconds** — past that the exchange has moved on and the advice is about a
point nobody is making any more. That is a factor of twenty or more, and it
will not come from prompt trimming.

The shape that could work is two tiers:

| tier | job | budget |
|---|---|---|
| fast | did that contain a fallacy, and what is the one-line reply | 3–5s |
| slow | the full treatment, running behind, for the debrief | as now |

The fast tier probably cannot be a frontier model over the network. A small
local model via Ollama is the obvious candidate, and its weaker schema-holding
— already noted in `providers/ollama.py` — matters much less when the output
is one sentence rather than a JSON record.

**Streaming ASR replaces batch.** Current transcription is whole-file: download
audio, run Whisper, get text. Live needs a microphone stream, incremental
decode, and a rolling window rather than 12,000-character chunks with overlap.
`faster-whisper` supports streaming; the chunking logic in `analysis.py` does
not, and should not be bent into it.

**Speaker attribution gets harder, and it is already the weak point.** Whisper
emits no speaker labels; the current reports infer attribution from context
and say so. Live, with two people interrupting each other, context inference
degrades badly. This needs real diarisation, which is a separate model and a
separate dependency.

## The part that is not an engineering question

Analysing a published YouTube video is commentary on material its speakers
made public. **Transcribing a live conversation is recording it.**

In the UK, recording a conversation you are part of is lawful for personal
use. Processing the other person's speech through a product is a different
question, and it reaches UK GDPR the moment it stops being purely personal —
which is exactly when it becomes a product with customers.

The interview-assistant tools that work this way generally handle it by being
explicitly one-sided and disclosed. Before any build:

- who is the data controller when a customer runs this
- is the other party's speech stored, or only transcribed and discarded
- what the disclosure requirement is in the jurisdictions being sold into
- whether a "notes only on my own speech" mode is the safer default product

This is a question for counsel, not for the engineering plan, and it should be
answered before rather than after — it may change what gets built.

## What it changes commercially

The current tool serves someone **preparing**. Live serves someone
**performing**. Different buyer, different tolerance for error, and a much
worse failure mode: a wrong fallacy call whispered mid-debate is worse than no
tool, because it is acted on immediately and in public.

## If it goes ahead, the first session is not code

1. Latency budget: prove a small local model can name a fallacy and draft a
   reply in under five seconds on the target hardware. If it cannot, stop.
2. Consent model: settle the questions above.
3. Only then: streaming ASR, rolling window, diarisation.

Steps 1 and 2 are each a day. Both can kill the feature, which is the point of
doing them first.

## Open questions

- Does it listen to both sides, or only the user's opponent?
- Is output on-screen, or audio in an earpiece? The second is a very
  different product and a very different set of norms.
- Does the debrief afterwards reuse the existing report format? Probably yes,
  and that is a real advantage — the live tier becomes an addition to a
  working product rather than a rewrite.
