\# AEMO Detector Input Processing and Synthetic-Only Tuning



\## Purpose



Initial experiments on raw AEMO TOTALDEMAND produced a large number of

detector activations. This work investigates whether high-frequency and

intraday demand variation contributes to this behaviour and defines a

reproducible preprocessing and detector-selection procedure.



The work covers three drift detectors:



\- ADWIN

\- KSWIN

\- Page-Hinkley



The two AEMO regions evaluated are SA1 and NSW1.



\## AEMO evaluation period



The existing frozen AEMO split is used:



\- Training: 2018-01-01 to 2019-12-31

\- Calibration: 2020-01-01 to 2020-02-29

\- Test stream: 2020-03-01 onward



Detector comparison in this experiment uses the frozen test stream.



The standardised test stream contains 67,249 30-minute TOTALDEMAND

observations for each region, ending at 2024-01-01 00:00.



\## Input processing



The raw detector input is the standardised 30-minute TOTALDEMAND series.



A daily aggregation is used as the first input-processing method:



1\. Resample TOTALDEMAND into one-day groups.

2\. Calculate the mean demand for each day.

3\. Retain only complete days containing 48 standardised 30-minute samples.

4\. Do not impute missing values.

5\. Do not remove negative demand observations before aggregation.



The final test stream contains 1,401 complete daily observations from

2020-03-01 through 2023-12-31.



Daily aggregation suppresses intraday/high-frequency variation before

the drift detectors are applied. It is treated as temporal aggregation

rather than proof that every removed detector activation was caused by

seasonality.



\## Parameter selection protocol



Detector parameters are tuned using synthetic datasets only.



The AEMO test stream is not used to select detector hyperparameters.



Synthetic experiments use:



\- Drift types: none, sudden, gradual, recurring

\- Samples per series: 20,000

\- Noise: 1.0

\- Seeds: 1, 2, 3, 4, 5



Each sensitivity sweep changes one detector parameter at a time while

holding the remaining parameters fixed.



The shared evaluation implementation provides:



\- Detection delay

\- False alarms per 10,000 samples

\- Missed detections



Synthetic false-alarm behaviour, missed detections, and detection delay

are used to select fixed detector configurations.



\## Frozen detector configurations



The configurations selected before final AEMO application are:



\### ADWIN



\- delta: 0.001



This setting retained detection of the synthetic drift events while

substantially reducing false alarms compared with more sensitive ADWIN

settings.



\### KSWIN



\- alpha: 0.005

\- window\_size: 300

\- stat\_size: 30

\- internal seed: 42



KSWIN showed a strong sensitivity-specificity trade-off.



Increasing stat\_size to 45 eliminated detections on the synthetic

no-drift series but missed gradual drift in most seeds. The selected

window\_size of 300 retained detection of sudden, gradual, and recurring

synthetic drift while reducing false-alarm activity relative to the

canonical window size of 100.



\### Page-Hinkley



\- min\_instances: 30

\- delta: 0.005

\- threshold: 200.0



Increasing the threshold to 200 substantially reduced synthetic

false-alarm activity while retaining detection of the tested drift

types.



\## AEMO results



The final detector configurations were fixed before being applied to

the processed AEMO test streams.



| Region | Detector | Raw 30-min | Daily baseline | Daily + frozen | Raw-to-final reduction |

| --- | --- | ---: | ---: | ---: | ---: |

| NSW1 | ADWIN | 742 | 13 | 13 | 98.2% |

| NSW1 | KSWIN | 580 | 13 | 4 | 99.3% |

| NSW1 | Page-Hinkley | 2241 | 46 | 46 | 97.9% |

| SA1 | ADWIN | 682 | 9 | 9 | 98.7% |

| SA1 | KSWIN | 599 | 12 | 4 | 99.3% |

| SA1 | Page-Hinkley | 2241 | 46 | 46 | 97.9% |



The large reduction from raw 30-minute input to daily input indicates

that high-frequency/intraday variation was responsible for substantial

detector activity.



Synthetic-only parameter selection produced an additional reduction for

KSWIN, from 12 to 4 detections in SA1 and from 13 to 4 detections in

NSW1.



ADWIN and Page-Hinkley produced the same detection counts before and

after application of their selected synthetic configurations on the

daily AEMO streams.



\## Interpretation limitation



AEMO detection counts are not labelled as false alarms in this

experiment.



Documented real-world events have not yet been matched against detector

changepoints, and the available event catalogue may not represent

complete ground truth. Therefore, unmatched AEMO detections must not

automatically be classified as false alarms.



A separate AEMO event-matching procedure will be used for subsequent

real-data evaluation.



\## Reproducibility



Experiment runs are recorded through the shared experiment harness in:



`results/runs.csv`



Detector configurations are recorded using configuration hashes and

their corresponding configuration JSON files.



The AEMO comparison is produced from recorded experiment results using:



`experiments/produce\_aemo\_processing\_results.py`



The producer does not rerun detectors or recompute detection metrics.



Its generated comparison is written to:



`results/aemo\_processing\_summary.csv`
