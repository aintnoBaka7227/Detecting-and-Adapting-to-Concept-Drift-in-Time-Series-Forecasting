# Event Tiering Criteria

How the documented AEMO events are tiered and used. `src/drift_lab/aemo/events.csv` is the source of truth: each event's tier is in its `tier` column and the reason in `tier_reason`.

`tests/test_docs_in_sync.py` fails if the Tier 1 list below stops matching the catalogue, or if this file names a path or function that no longer exists.

## Tier 1

An event is Tier 1 when it is:

- large or substantial in market or system relevance;
- documented in an AEMO or AER source; and
- tied to a clear expected mechanism affecting demand, price, market behaviour or system operation.

The Tier 1 events are:

| Event ID | Event |
|---|---|
| `EVT-2020-02` | NEM COVID-19 demand change |
| `EVT-2020-07` | NEM December solar and mild-weather demand erosion |
| `EVT-2022-03` | NEM spot market suspension |
| `EVT-2023-14` | SA and NSW system-security directions |

## Tier 2

Tier 2 holds every other catalogue event. These are genuine reference events that are smaller, more regional, shorter or more ambiguous than Tier 1. Events are kept for that reason, not removed.

## Matching

`match_unmatch()` in `src/drift_lab/evaluation/evaluation.py` applies these steps:

1. **Region:** use the region's events plus every `NEM` event.
2. **Refractory:** keep the earliest detection, and label later ones within 14 days `Ignored`. Matching ignores these, and refractory filtering does not depend on events.
3. **Order:** go through events by tier, then start date, then event ID.
4. **Match:** each event takes the first unused detection in its window, so a detection matches at most one event and an event is credited at most once.
5. **Label:** detections left over are `Unmatch`.

`calculate_match_window()` sets each window, left-inclusive and right-exclusive:

| Event `date_precision` | Window |
|---|---|
| `day` | `[start, start + 1 day + 7 days)` |
| Any other | `[start, end + 1 day + 7 days)` |

A detection before an event's start never matches it. Only TEST alarms are matched, so an event before TEST cannot be covered.

The catalogue is contextual, not complete ground truth. A detection that matches no event is `unmatched`, not a false alarm.

## Freeze rule

Tiers are set independently of detector outputs. Detector locations or matching results must not be used to add, promote, demote or change an event. A factual correction after the freeze must be made in git with a clear justification.
