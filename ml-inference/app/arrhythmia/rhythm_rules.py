def _instantaneous_hr(times):
    hr = [0.0] * len(times)
    for i in range(1, len(times)):
        rr = times[i] - times[i - 1]
        hr[i] = 60.0 / rr if rr > 0 else 0.0
    if len(hr) > 1:
        hr[0] = hr[1]
    return hr


def _vt_alerts(times, classes):
    """A run breaks whenever a non-V beat appears, or the gap to the previous
    V beat is >= 0.60s (a fast-run tachycardia requires both)."""
    alerts = []
    run_indices = []

    def close_run():
        if len(run_indices) >= 3:
            alerts.append({
                "type": "VT",
                "severity": "critical",
                "start": times[run_indices[0]],
                "end": times[run_indices[-1]],
            })

    for i, cls in enumerate(classes):
        if cls == "V":
            if run_indices and (times[i] - times[run_indices[-1]]) < 0.60:
                run_indices.append(i)
            else:
                close_run()
                run_indices = [i]
        else:
            close_run()
            run_indices = []

    close_run()
    return alerts


def _bigeminy_alerts(times, classes):
    alerts = []
    n = len(classes)
    i = 0
    while i + 5 < n:
        if classes[i:i + 6] == ["N", "V", "N", "V", "N", "V"]:
            alerts.append({"type": "BIGEMINY", "severity": "warning", "start": times[i], "end": times[i + 5]})
            i += 6
        else:
            i += 1
    return alerts


def _trigeminy_alerts(times, classes):
    alerts = []
    n = len(classes)
    pattern = ["N", "N", "V"] * 3
    i = 0
    while i + 8 < n:
        if classes[i:i + 9] == pattern:
            alerts.append({"type": "TRIGEMINY", "severity": "warning", "start": times[i], "end": times[i + 8]})
            i += 9
        else:
            i += 1
    return alerts


def _rate_alerts(times, window_sec=10.0):
    alerts = []
    n = len(times)
    if n < 2:
        return alerts

    hrs = _instantaneous_hr(times)
    for i in range(n):
        t = times[i]
        j = i
        while j > 0 and times[j - 1] >= t - window_sec:
            j -= 1
        # allow near-full windows: discrete beat spacing rarely aligns exactly with window_sec
        if t - times[j] < window_sec * 0.9:
            continue

        window_hrs = hrs[j:i + 1]
        mean_hr = sum(window_hrs) / len(window_hrs)

        if mean_hr > 150:
            alerts.append({"type": "TACHY", "severity": "critical", "start": times[j], "end": t, "mean_hr": round(mean_hr, 1)})
        elif mean_hr > 100:
            alerts.append({"type": "TACHY", "severity": "warning", "start": times[j], "end": t, "mean_hr": round(mean_hr, 1)})
        elif mean_hr < 50:
            alerts.append({"type": "BRADY", "severity": "warning", "start": times[j], "end": t, "mean_hr": round(mean_hr, 1)})

    return alerts


def _afib_alerts(times, window_beats=10, cv_threshold=0.15):
    alerts = []
    if len(times) < window_beats + 1:
        return alerts

    rrs = [times[i] - times[i - 1] for i in range(1, len(times))]
    positive_streak = 0
    streak_start = None

    for i in range(len(rrs) - window_beats + 1):
        segment = rrs[i:i + window_beats]
        mean_rr = sum(segment) / len(segment)
        variance = sum((x - mean_rr) ** 2 for x in segment) / len(segment)
        cv = (variance ** 0.5) / mean_rr if mean_rr > 0 else 0.0

        if cv > cv_threshold:
            if positive_streak == 0:
                streak_start = times[i + 1]
            positive_streak += 1
            if positive_streak >= 2:
                alerts.append({"type": "AFIB", "severity": "warning", "start": streak_start, "end": times[i + window_beats]})
        else:
            positive_streak = 0
            streak_start = None

    return alerts


def _merge_alerts(alerts, merge_window_sec=5.0):
    if not alerts:
        return []

    alerts = sorted(alerts, key=lambda a: (a["type"], a["start"]))
    merged = [dict(alerts[0])]

    for alert in alerts[1:]:
        last = merged[-1]
        if alert["type"] == last["type"] and alert["start"] <= last["end"] + merge_window_sec:
            last["end"] = max(last["end"], alert["end"])
        else:
            merged.append(dict(alert))

    return sorted(merged, key=lambda a: a["start"])


def detect_alerts(beats: list[dict]) -> list[dict]:
    """beats: list of {"time": float, "class": str}, sorted by time."""
    times = [b["time"] for b in beats]
    classes = [b["class"] for b in beats]

    alerts = []
    alerts += _vt_alerts(times, classes)
    alerts += _bigeminy_alerts(times, classes)
    alerts += _trigeminy_alerts(times, classes)
    alerts += _rate_alerts(times)
    alerts += _afib_alerts(times)

    return _merge_alerts(alerts)
