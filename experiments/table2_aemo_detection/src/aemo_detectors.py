
from river import drift


def create_detector(name, params=None):

    if params is None:
        params = {}

    params = params.copy()

    if name == "ADWIN":
        detector = drift.ADWIN(**params)

    elif name == "KSWIN":
        # Use the same random seed for KSWIN
        if "seed" not in params:
            params["seed"] = 42

        detector = drift.KSWIN(**params)

    elif name == "Page-Hinkley":
        detector = drift.PageHinkley(**params)

    else:
        raise ValueError("Unknown detector: " + name)

    return detector


def run_detector(values, name, params=None):

    detector = create_detector(name, params)

    alarms = []

    for i, value in enumerate(values):

        detector.update(float(value))

        if detector.drift_detected:
            alarms.append(i)

    return alarms