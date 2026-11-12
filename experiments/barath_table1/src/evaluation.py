
def evaluate_detections(
    alarms,
    true_drifts,
    n,
    matching_window=336,
    drift_type="sudden"
):

    alarms = sorted(set(int(a) for a in alarms))
    true_drifts = sorted(int(t) for t in true_drifts)

    # Gradual drift has a start and end
    if drift_type == "gradual":

        if len(true_drifts) != 2:
            raise ValueError(
                "Gradual drift needs a start and end point"
            )

        events = [(true_drifts[0], true_drifts[1])]

    else:
        # Sudden and recurring drift use change points
        events = [(cp, cp) for cp in true_drifts]

    matched_alarms = set()
    delays = []
    missed = 0

    for start, end in events:

        candidates = [
            alarm for alarm in alarms
            if start <= alarm <= end + matching_window
            and alarm not in matched_alarms
        ]

        if candidates:
            first_alarm = candidates[0]
            matched_alarms.add(first_alarm)

            delays.append(first_alarm - start)

        else:
            missed += 1

    false_alarms = len(alarms) - len(matched_alarms)

    false_alarms_per_10k = (
        false_alarms / n * 10000
        if n > 0 else 0
    )

    mean_delay = (
        sum(delays) / len(delays)
        if delays else None
    )

    return {
        "mean_delay": mean_delay,
        "false_alarms": false_alarms,
        "false_alarms_per_10k": false_alarms_per_10k,
        "missed_detections": missed,
        "matched_detections": len(matched_alarms),
        "delays": delays
    }
