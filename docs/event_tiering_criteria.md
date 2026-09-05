# Event Tiering Criteria

## Purpose

This document defines the event-tiering procedure for the documented AEMO event catalogue (`data/events.csv`). The tier assignments are fixed before real-data detector-event matching so that detector outputs do not influence which events are treated as primary reference events.

## Event Catalogue

All documented events are retained in `events.csv`. Events are not removed because they are smaller, regional, short-duration, or ambiguous. Instead, each event is assigned to Tier 1 or Tier 2.

## Tier 1

Tier 1 contains five primary reference events. An event is assigned to Tier 1 when it is:

- large or substantial in market or system relevance;
- documented in an AEMO or AER source; and
- associated with a clear expected mechanism affecting electricity demand, price or market behaviour, or system operation.

The five Tier 1 events are:

1. `EVT-2020-02` — NEM COVID-19 demand change
2. `EVT-2020-07` — NEM December solar and mild-weather demand erosion
3. `EVT-2021-10` — Five-Minute Settlement commencement
4. `EVT-2022-03` — NEM spot market suspension
5. `EVT-2023-14` — SA and NSW system-security directions

## Tier 2

Tier 2 contains all other documented events in the catalogue. These are genuine contextual events but may be smaller, more regional, shorter in duration, or more ambiguous than the Tier 1 reference events.

## Pre-Specification Rule

Tier assignments are determined independently of detector outputs and are frozen before real-data detector-event matching. Detector locations or subsequent matching results must not be used to select, promote, demote, or otherwise change event tiers.

If a factual correction is required after the catalogue is frozen, the change should be documented through version control with a clear justification.

## Use During Detector Matching

Real-data detections are checked against Tier 1 events first. A detection that does not match a Tier 1 event is then checked against the Tier 2 catalogue before being reported as unmatched.

The event catalogue is contextual rather than exhaustive ground truth. Therefore, a real-data detection that matches neither tier should be reported as unmatched rather than automatically classified as a false alarm.

The temporal matching rule used to associate detections with documented events must be defined before examining detector-event matches.

## Reproducibility

The documented event catalogue and tiering criteria are version-controlled in Git before detector-event matching. The detector-event matching rule and detector configurations will also be recorded before the real-data detection analysis. This ensures that the event classifications and analysis settings can be traced and reproduced.
