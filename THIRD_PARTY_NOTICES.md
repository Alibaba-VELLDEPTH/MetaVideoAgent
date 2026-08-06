# Third-Party Notices

MetaVideoAgent does not vendor third-party source code, model weights, or media
assets. Python dependencies are downloaded separately by the package installer
and remain subject to their respective upstream licenses. The source repository
does include the selected CG-Bench annotation fields described below.

The core dependency set includes ChromaDB, HTTPX, imageio-ffmpeg, the OpenAI
Python client, OpenCV, and Pillow. Consult each installed
distribution's metadata and upstream repository for the license that applies to
the exact version selected in your environment.

Codex CLI, or any compatible coding-agent wrapper selected by the user, is an
external executable and is not distributed as part of this repository. FFmpeg
is obtained through `imageio-ffmpeg` when available or from the user's system
installation; its own license applies. Model services,
datasets, videos, and generated artifacts may carry
separate terms. Users are responsible for ensuring that their selected provider,
models, datasets, and media may be used for their intended purpose.

## VA-EvoBench and CG-Bench

The source repository includes `VA-EvoBench/`, which contains selected
annotation fields derived from CG-Bench together with original VA-EvoBench
curation metadata and utility scripts. The repository's Apache License 2.0
covers the original VA-EvoBench contributions but does not relicense CG-Bench
annotations or media. No CG-Bench video, audio, or frame content is bundled.
The upstream license, gated-access agreement, citation requirements, and
original media rights continue to apply; see `VA-EvoBench/NOTICE.md`.
