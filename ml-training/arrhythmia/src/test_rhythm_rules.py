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
