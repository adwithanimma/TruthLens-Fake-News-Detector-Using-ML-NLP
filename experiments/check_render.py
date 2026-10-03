import os
import sys

import requests

B = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("TRUTHLENS_URL", "http://127.0.0.1:5000")
    ).rstrip("/")


def pct(value):
    """Identical to pct() in static/js/app.js."""
    return str(round(value * 1000) / 10) + "%"


TEXTS = {
    "real-ish": (
        "The transport authority said all services would return to normal by Friday "
        "after engineers replaced damaged track near the estuary. Commuters faced "
        "delays of up to forty minutes while crews worked overnight on the junction "
        "outside the harbour. The agency said the inspection found no structural "
        "concerns with the bridge itself."
    ),
    "fake-ish": (
        "Scientists confirm Earth is definitively flat and every photo from the space "
        "station was staged, according to a viral post. The post claims a whistleblower "
        "engineer proved the curvature models were invented to mislead the public, and "
        "it urges followers to ignore weather satellites because clouds prove it."
    ),
    "neutral": (
        "Shares closed higher on Tuesday as investors weighed a quarterly earnings "
        "report against commentary from the central bank. The index gained about one "
        "percent, while bond yields were little changed. Analysts described trading "
        "as orderly and said volumes were in line with the recent average."
    ),
}

failures = []

try:
    requests.get(B + "/api/health", timeout=5).raise_for_status()
except requests.RequestException as exc:
    print(f"Cannot reach {B}: {exc}")
    print("Start the server with `python app.py` and use the port it reports.")
    raise SystemExit(2)

for name, text in TEXTS.items():
    r = requests.post(B + "/api/predict", json={"text": text}, timeout=20)
    r.raise_for_status()
    d = r.json()["result"]
    real = d["scores"]["real"]
    fake = d["scores"]["fake"]

    shown_real = pct(real)
    shown_fake = pct(fake)
    shown_conf = pct(d["confidence"])
    total = float(shown_real[:-1]) + float(shown_fake[:-1])

    print(
        f"{name:9s} label={d['label']:4s} conf={shown_conf:>6s} "
        f"real={shown_real:>6s} fake={shown_fake:>6s} sum={total:.1f}%"
    )

    if any("%" in v for v in (shown_real, shown_fake, shown_conf)):
        pass
    if float(shown_conf[:-1]) > 100.0001:
        failures.append(f"{name}: confidence over 100% ({shown_conf})")
    if abs(total - 100.0) > 0.15:
        failures.append(f"{name}: real+fake = {total:.1f}%, expected ~100%")
    if float(shown_real[:-1]) > 100.0001 or float(shown_fake[:-1]) > 100.0001:
        failures.append(f"{name}: a probability bar exceeded 100%")
    if abs(d["confidence"] - max(real, fake)) > 0.011:
        failures.append(f"{name}: confidence does not match the larger score")

print()
if failures:
    for f in failures:
        print("FAIL:", f)
    raise SystemExit(1)
print("PASS: all rendered percentages are in range and internally consistent")
