# Notice, provenance, and upstream conditions

This benchmark directory contains two kinds of material:

1. selected CG-Bench QA annotation fields, including questions, choices,
   answers, source categories, and temporal evidence intervals; and
2. VA-EvoBench curation metadata, including target-distribution assignments,
   video-disjoint split membership, stable instance identifiers, unified field
   names, and global-review timestamps.

No video, audio, frame image, transcript file, model output, or execution trace
is redistributed.

CG-Bench is the upstream source. Its
[official dataset page](https://huggingface.co/datasets/CG-Bench/CG-Bench) identifies the
annotation release as MIT-licensed and additionally requires users to accept a
gated access agreement. That agreement states that the dataset must not be used
for experiments that harm human subjects, that other agreements may apply, and
that video copyrights remain with the original creators or platforms for
academic research use. Those upstream conditions are not replaced, relaxed, or
expanded by this package.

To access media, use the official CG-Bench distribution cited above and accept
its gated access conditions. VA-EvoBench does not mirror or extend the upstream
media distribution.

The package contains no contributor identity, email address, account name,
credential, private storage location, signed media URL, or institution-specific
experiment path. Some source questions and choices mention people or visible
web addresses that occur in the original public-facing video content. These
are upstream annotation content and must not be repurposed for identity
profiling, surveillance, or harmful use.

Use of this benchmark must preserve the annotation-access rules in `README.md`
and must accurately disclose any protocol deviation.

Original VA-EvoBench curation metadata, split definitions, identifiers, and
utility scripts are made available under the MetaVideoAgent repository's
Apache License 2.0. Selected CG-Bench annotation fields are redistributed under
the upstream CG-Bench license and gated-access conditions. The Apache License
does not relicense CG-Bench annotations, videos, or any other third-party
material. Users should cite both VA-EvoBench/MetaVideoAgent and CG-Bench in
publications based on this benchmark.
