from rhythm_rules import detect_alerts


def test_three_v_beats_fast_gives_vt_critical():
    beats = [{"time": i * 0.5, "class": "V"} for i in range(3)]
    alerts = detect_alerts(beats)
    assert any(a["type"] == "VT" and a["severity"] == "critical" for a in alerts)


def test_two_v_beats_gives_no_vt():
    beats = [{"time": i * 0.5, "class": "V"} for i in range(2)]
    alerts = detect_alerts(beats)
    assert not any(a["type"] == "VT" for a in alerts)


def test_bigeminy_pattern():
    seq = ["N", "V", "N", "V", "N", "V"]
    beats = [{"time": i * 0.8, "class": c} for i, c in enumerate(seq)]
    alerts = detect_alerts(beats)
    assert any(a["type"] == "BIGEMINY" for a in alerts)


def test_trigeminy_pattern():
    seq = ["N", "N", "V"] * 3
    beats = [{"time": i * 0.8, "class": c} for i, c in enumerate(seq)]
    alerts = detect_alerts(beats)
    assert any(a["type"] == "TRIGEMINY" for a in alerts)


def test_bradycardia_sustained():
    rr = 60 / 45.0
    n_beats = int(12 / rr) + 2
    beats = [{"time": i * rr, "class": "N"} for i in range(n_beats)]
    alerts = detect_alerts(beats)
    assert any(a["type"] == "BRADY" for a in alerts)


def test_alerts_merge_within_5s():
    beats = [{"time": i * 0.5, "class": "V"} for i in range(3)]
    beats += [{"time": 3 * 0.5 + 3.0 + i * 0.5, "class": "V"} for i in range(3)]
    alerts = detect_alerts(beats)
    vt_alerts = [a for a in alerts if a["type"] == "VT"]
    assert len(vt_alerts) == 1


def test_alerts_do_not_merge_beyond_5s():
    beats = [{"time": i * 0.5, "class": "V"} for i in range(3)]
    beats += [{"time": 3 * 0.5 + 10.0 + i * 0.5, "class": "V"} for i in range(3)]
    alerts = detect_alerts(beats)
    vt_alerts = [a for a in alerts if a["type"] == "VT"]
    assert len(vt_alerts) == 2


def _beats(times, v_idx=()):
    return [{"time": t, "class": "V" if i in v_idx else "N"} for i, t in enumerate(times)]


def test_pvcs_in_regular_sinus_rhythm_are_not_afib():
    v_idx = {6, 14, 22}
    times = [0.85 * i - (0.25 if i in v_idx else 0.0) for i in range(32)]  # premature V, then compensatory pause
    types = [a["type"] for a in detect_alerts(_beats(times, v_idx))]
    assert "AFIB" not in types


def test_irregularly_irregular_normal_beats_still_flag_afib():
    import random

    random.seed(3)
    t, times = 0.0, []
    for _ in range(40):
        t += random.uniform(0.45, 1.25)
        times.append(t)
    assert "AFIB" in [a["type"] for a in detect_alerts(_beats(times))]
