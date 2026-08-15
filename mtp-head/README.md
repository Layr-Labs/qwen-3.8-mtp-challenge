This directory is intentionally a placeholder.

The MTP head for this submission is the ORGANIZER-PINNED head: see
../mtp-head.manifest.json, which declares `"source": "pinned"`. With a pinned
source the runner fetches and digest-verifies the head out of band
(EigenLabs/Qwen3.8-27B-MTP-bf16) and never reads in-branch weights from this
directory, so no head weights are shipped here.

This file exists only so `yukon submit`'s archive step (which tars every
editablePaths entry) does not fail on a missing optional directory. The path is
exempt from the editable-surface byte budget and excluded from static review,
so this placeholder is inert for scoring.
