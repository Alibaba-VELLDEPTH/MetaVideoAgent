# VA-EvoBench Dataset Card

## Summary

VA-EvoBench evaluates whether a video agent can adapt its functional design to
a coherent target video distribution. It defines a separate evolution problem
for each distribution rather than jointly evolving one agent on a heterogeneous
mixture. The benchmark is released with MetaVideoAgent and is derived from
CG-Bench.

## Composition

| Target distribution | Evolution videos | Evolution questions | Held-out videos | Held-out questions |
|---|---:|---:|---:|---:|
| Interview | 5 | 41 | 14 | 135 |
| Product Presentation | 5 | 50 | 12 | 103 |
| Software Tutorial | 5 | 50 | 14 | 132 |
| Dramatic Narrative | 5 | 50 | 12 | 119 |
| Stage Performance | 5 | 42 | 11 | 103 |
| Gameplay | 5 | 43 | 9 | 85 |
| Sports Broadcast | 4 | 44 | 8 | 71 |
| Course Lecture | 4 | 36 | 12 | 119 |
| **Total** | **38** | **356** | **92** | **867** |

The split unit is `video_id`. No video occurs in both splits or in multiple
target distributions. All questions belonging to one selected video remain in
the same split.

## Source and curation

Videos and multiple-choice QA annotations originate from CG-Bench. Candidate
videos were manually grouped by recurring content form, modality composition,
and evidence-localization requirements. Curators inspected a global overview
of ten uniformly spaced frames across each candidate video before assigning it
to a target distribution. The JSONL files define the exact video and question
membership used in the reported experiments; no undisclosed filtering is
applied at evaluation time.

The benchmark contains the following target distributions:

| Distribution | Content form | Recurring evidence requirements |
|---|---|---|
| Interview | Interviews and dialogue-centered interactions | Spoken content, speaker identity, and local visual interactions |
| Product Presentation | Product demonstrations and sales presentations | Product attributes, displayed text, orientation, and relative positions |
| Software Tutorial | Screen-based software instruction | Interface text, operation order, and state changes |
| Dramatic Narrative | TV-drama and narrative scenes | Characters, events, and relations across cuts |
| Stage Performance | Magic and other stage interactions | Event phases, ordinal relations, and brief actions |
| Gameplay | Game recordings | Dynamic scene state, player/object actions, and interface evidence |
| Sports Broadcast | Sports events and broadcasts | Subject tracking across views, actions, scores, and localized events |
| Course Lecture | Instructional lectures with slides | Lecture content, on-screen text, concepts, and page--speech timing |

These descriptions characterize recurring evidence requirements. They are not
question-type routing rules and do not prescribe a fixed agent architecture.

## Annotation provenance

Question text, choices, textual answers, source domains, source question
categories, and temporal evidence intervals are selected CG-Bench annotation
fields. VA-EvoBench contributes the target-distribution assignment, the
video-disjoint evolution/held-out protocol, exact membership, stable instance
identifiers, unified field names, and review timestamps. Source temporal
annotations are serialized uniformly as `time_reference` in seconds.

## Intended use

VA-EvoBench is intended for research on video understanding, video-agent
adaptation, agent evolution, evidence localization, and controlled evaluation
under target-distribution shift. Fair comparison requires:

1. one independent evolution run per target distribution;
2. no use of held-out annotations or outcomes during evolution;
3. reporting the last-iteration agent under a declared evolution budget;
4. reporting per-distribution accuracy and their unweighted macro average; and
5. clearly documenting any deviation from the supplied inputs or protocol.

The benchmark is not intended for identifying individuals, inferring sensitive
personal attributes, surveillance, face recognition, or any experiment that
could harm human subjects. It is not a training corpus for republishing video
content or bypassing upstream access conditions.

## Annotation access and evaluation integrity

`answer`, `correct_choice`, and `time_reference` are distributed for evaluation
and audit. A compliant Student must not access them. Evolution-split labels
and temporal references are available only to post-execution Teacher review.
Held-out annotations and outcomes are reporting-only and must not influence any
adaptation decision. Because the public benchmark necessarily exposes held-out
labels, benchmark users are responsible for enforcing this separation in their
evaluation environment and documenting it in reported results.

## Privacy and sensitive content

The package adds no participant records, contact information, accounts,
credentials, local paths, execution logs, or author metadata. It contains no
video or frame pixels. Source questions can mention people, public-facing names,
spoken statements, or visible on-screen text because these are part of the
upstream video-QA annotations. Such content must be handled under the upstream
CG-Bench conditions and must not be repurposed for identity profiling or harm.

## Media, rights, and access

The package does not redistribute CG-Bench videos. Users must obtain media
through the [official gated CG-Bench dataset](https://huggingface.co/datasets/CG-Bench/CG-Bench), accept
the upstream access agreement, and comply with the original creators' and
platforms' rights. See `NOTICE.md` for the exact scope of this package.

## Limitations

- The eight distributions are curated from CG-Bench and do not exhaust all
  real-world video distributions.
- Category boundaries reflect recurring evidence requirements and may not
  capture every property of an individual video.
- Multiple-choice accuracy does not by itself verify evidence grounding.
- Public held-out labels make procedural separation, rather than cryptographic
  hiding, responsible for held-out integrity.
- Video availability and permitted uses depend on the upstream release and may
  change independently of this package.
