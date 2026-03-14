# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8")

"""
Test: How the system handles changing counter values.

Simulates what happens when the background poller reads the counter
over time — normal increments, impossible jumps, backwards reads, low confidence.

Run with:
    cd backend
    python test_value_changes.py
"""

from datetime import datetime, timedelta
from app.utils.counter_validator import validate_reading


def check(label, new_value, new_confidence, prev_value, prev_timestamp, current_timestamp):
    result = validate_reading(
        new_value=new_value,
        new_confidence=new_confidence,
        prev_value=prev_value,
        prev_timestamp=prev_timestamp,
        current_timestamp=current_timestamp,
        min_confidence=0.85,
        max_rate_per_second=5.0,
    )
    status = "[ACCEPTED]" if result.is_valid else "[REJECTED]"
    print(f"\n{status}  |  {label}")
    print(f"   Value:      {prev_value} -> {new_value}")
    print(f"   Confidence: {new_confidence}")
    print(f"   Reason:     {result.reason}")
    return result


print("=" * 65)
print("  COUNTER VALIDATION — SIMULATED VALUE CHANGE TESTS")
print("=" * 65)

# Base timestamp
t0 = datetime(2026, 3, 12, 10, 0, 0)

# ── Test 1: First ever reading (no history) ───────────────────
check(
    "First reading — no previous value",
    new_value=18900, new_confidence=0.96,
    prev_value=None, prev_timestamp=None,
    current_timestamp=t0,
)

# ── Test 2: Normal increment (1 second later, +5 counts) ─────
check(
    "Normal increment — +5 in 1 second (5.0/sec, within limit)",
    new_value=18905, new_confidence=0.94,
    prev_value=18900, prev_timestamp=t0,
    current_timestamp=t0 + timedelta(seconds=1),
)

# ── Test 3: Normal increment (3 seconds later, +12 counts) ───
check(
    "Normal increment — +12 in 3 seconds (4.0/sec, within limit)",
    new_value=18917, new_confidence=0.97,
    prev_value=18905, prev_timestamp=t0 + timedelta(seconds=1),
    current_timestamp=t0 + timedelta(seconds=4),
)

# ── Test 4: Low confidence — digit mid-transition ─────────────
check(
    "Low confidence — digit rolling/blurry (confidence 0.61)",
    new_value=18920, new_confidence=0.61,
    prev_value=18917, prev_timestamp=t0 + timedelta(seconds=4),
    current_timestamp=t0 + timedelta(seconds=4.5),
)

# ── Test 5: Impossible jump — OCR misread ─────────────────────
check(
    "Impossible jump — +5000 in 1 second (OCR misread)",
    new_value=23917, new_confidence=0.91,
    prev_value=18917, prev_timestamp=t0 + timedelta(seconds=4),
    current_timestamp=t0 + timedelta(seconds=5),
)

# ── Test 6: Counter went backwards — misread digit ────────────
check(
    "Counter backwards — 18917 → 18817 (misread 9 as 8)",
    new_value=18817, new_confidence=0.88,
    prev_value=18917, prev_timestamp=t0 + timedelta(seconds=4),
    current_timestamp=t0 + timedelta(seconds=6),
)

# ── Test 7: Job reset — counter back to near zero ─────────────
check(
    "Job reset — counter went from 18917 to 12 (new job started)",
    new_value=12, new_confidence=0.95,
    prev_value=18917, prev_timestamp=t0 + timedelta(seconds=4),
    current_timestamp=t0 + timedelta(seconds=120),
)

# ── Test 8: Machine idle — no change in 30 seconds ───────────
check(
    "Machine idle — same value after 30 seconds (+0, still valid)",
    new_value=18917, new_confidence=0.98,
    prev_value=18917, prev_timestamp=t0 + timedelta(seconds=4),
    current_timestamp=t0 + timedelta(seconds=34),
)

print("\n" + "=" * 65)
print("  SHARPNESS GATE TEST — using actual image")
print("=" * 65)

import cv2
import numpy as np
import sys
import os

# Find any uploaded image in the project
img_path = None
for root, _, files in os.walk(os.path.dirname(__file__)):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, f)
            break
    if img_path:
        break

if img_path:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    print(f"\nImage: {os.path.basename(img_path)}")
    print(f"Sharpness score: {sharp_score:.1f}  (threshold = 80.0)")
    if sharp_score >= 80:
        print("[PASS] Image is sharp -- sharpness gate would PASS, OCR runs")
    else:
        print("[FAIL] Image is blurry -- sharpness gate would REJECT, OCR skipped")

    # Simulate a blurry frame (heavy Gaussian blur)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    blurry_score = cv2.Laplacian(blurred, cv2.CV_64F).var()
    print(f"\nSame image with heavy blur applied:")
    print(f"Sharpness score: {blurry_score:.1f}")
    verdict = "PASS" if blurry_score >= 80 else "REJECT -- digit mid-transition would be caught"
    print(f"[{verdict}]")
else:
    print("\nNo image found — upload an image via Swagger to test sharpness gate.")

print("\n" + "=" * 65)
