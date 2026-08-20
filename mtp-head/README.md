# mtp-head/ — optional in-branch MTP head weights

This directory is an OPTIONAL editable path (`benchmark.json`
`optionalEditablePaths`). It carries MTP head weights only when a submission
declares `"source": "in_branch"` in `../mtp-head.manifest.json`. The checked-in
declaration selects `"source": "pinned"` — the organizer-pinned head
(EigenLabs/Qwen3.8-27B-MTP-bf16), fetched and digest-verified by the runner out
of band — and with a pinned or remote source the runner NEVER reads weights
from this directory.

## Why this README is checked in

Submission archive tooling tars every `editablePaths` entry and fails on a path
that does not exist on disk, even an optional one (`yukon submit`:
`ENOENT ... lstat '.../mtp-head'`). This file keeps the directory present so
archiving succeeds while shipping no weights. It is inert for scoring by
construction, four times over: a top-level `README.md` is EXCLUDED from the
head tree digest (rule below), the path is exempt from the editable-surface
byte budget (`editableSurfaceByteBudget.exemptPaths`), it is excluded from the
static-review payload, and a pinned/remote source never consults the directory
at all. Keep this file in place — including alongside in-branch weights, where
the digest exclusion makes it invisible to verification.

## Declaring your own head in this directory

Set `../mtp-head.manifest.json` to `"source": "in_branch"` with a `path` under
`mtp-head/`, plus the `sha256` and `bytes` of what you ship (both required and
enforced; the runner refuses on mismatch or above `max_bytes`, 2 GiB).

TREE DIGEST RULE (mirrored from the trusted CLI's provenance sealing, so you
can recompute the number you are asked to declare): SHA-256 over the
concatenation, in `LC_ALL=C` sorted relative-path order, of
`"<hex file sha256>  <relative path>\n"` for every regular file in the tree
except a top-level `README.md`. `bytes` is the byte total of the same file set.
Equivalent shell, run inside the head directory:

```sh
find . -type f ! -name README.md \
  | sed 's|^\./||' \
  | LC_ALL=C sort \
  | while read -r f; do
      printf '%s  %s\n' "$(shasum -a 256 "$f" | awk '{print $1}')" "$f"
    done | shasum -a 256
```

A head only PROPOSES tokens — the organizer-pinned target decides every emitted
token, and the baseline leg of every ranked pair runs the pinned head
regardless of what this directory or the declaration says.
