from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ValidationResult:
    """Result of validating a new OCR counter reading."""
    is_valid: bool
    confidence_ok: bool        # confidence passed the threshold
    rate_ok: bool              # value change is physically possible
    direction_ok: bool         # counter didn't go backwards unexpectedly
    reason: str                # human-readable explanation

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "confidence_ok": self.confidence_ok,
            "rate_ok": self.rate_ok,
            "direction_ok": self.direction_ok,
            "reason": self.reason,
        }


def validate_reading(
    new_value: int,
    new_confidence: float,
    prev_value: Optional[int],
    prev_timestamp: Optional[datetime],
    current_timestamp: datetime,
    min_confidence: float = 0.85,
    max_rate_per_second: float = 5.0,
) -> ValidationResult:
    """
    Validate a new OCR counter reading against confidence and physical rate limits.

    Two checks are applied:

    1. Confidence gate
       The OCR confidence must be >= min_confidence.
       Below this, the reading is too uncertain to trust — mid-transition digits,
       glare, or blur can produce plausible-looking but wrong numbers.

    2. Rate validation
       The counter change from the previous accepted reading must be physically
       possible given the time elapsed and the machine's max speed.

       Formula:
           max_allowed_change = max_rate_per_second × elapsed_seconds × 1.2  (20% buffer)
           If new_value - prev_value > max_allowed_change → reject

       Also catches counter going backwards:
           If new_value < prev_value → rejected as impossible
           (counters only reset to 0 — a drop by any other amount = bad OCR read)

    Args:
        new_value:            The OCR-read counter value to validate.
        new_confidence:       OCR confidence score (0.0–1.0).
        prev_value:           Last accepted counter value (None if no history yet).
        prev_timestamp:       When prev_value was accepted (None if no history).
        current_timestamp:    Timestamp of the new reading.
        min_confidence:       Minimum acceptable confidence (default 0.85).
        max_rate_per_second:  Machine max sheets per second (default 5.0).
                              RMGT 11,000 SPH ÷ 3600 = 3.05/sec → use 5.0 with buffer.

    Returns:
        ValidationResult with is_valid=True if both checks pass.
    """

    # ── Check 1: Confidence gate ──────────────────────────────────────
    confidence_ok = new_confidence >= min_confidence

    if not confidence_ok:
        return ValidationResult(
            is_valid=False,
            confidence_ok=False,
            rate_ok=True,
            direction_ok=True,
            reason=(
                f"Confidence {new_confidence:.3f} is below threshold {min_confidence}. "
                f"Reading rejected — digit may be mid-transition or obscured."
            ),
        )

    # ── No previous reading — nothing to compare against ─────────────
    if prev_value is None or prev_timestamp is None:
        return ValidationResult(
            is_valid=True,
            confidence_ok=True,
            rate_ok=True,
            direction_ok=True,
            reason="First reading — no rate comparison available. Accepted.",
        )

    elapsed_seconds = (current_timestamp - prev_timestamp).total_seconds()
    delta = new_value - prev_value

    # ── Check 2a: Direction — counter must not go backwards ───────────
    # Exceptions:
    #   1. Counter reset to near-0 (job reset): prev=18900, new=12 → allow
    #   2. Large backwards jump (> 50): video loop or job change → accept as new baseline
    #   3. Small backwards delta (OCR noise tolerance ±10): last digit misread → accept
    is_reset = (new_value < 100 and prev_value > 1000) or (delta < -50)
    noise_tolerance = 20  # OCR can misread last 1-2 digits (e.g. 18929→18917 is noise/loop)
    direction_ok = delta >= -noise_tolerance or is_reset

    if not direction_ok:
        return ValidationResult(
            is_valid=False,
            confidence_ok=True,
            rate_ok=True,
            direction_ok=False,
            reason=(
                f"Counter went backwards: {prev_value} → {new_value} (delta={delta}). "
                f"Rejected — likely a bad OCR read of a digit (noise tolerance={noise_tolerance})."
            ),
        )

    # ── Check 2b: Rate — change must be physically possible ──────────
    # Skip rate check if we just came from a reset/near-zero baseline.
    # When counter resets to 0 (or OCR briefly reads near-0 on a garbled frame),
    # the next real reading will look like a huge jump from ~0 → 18917, which is
    # physically fine — the machine was already at 18917, we just lost the baseline.
    coming_from_reset = prev_value < 100
    max_allowed = max_rate_per_second * elapsed_seconds * 1.2 if elapsed_seconds > 0 else 0
    # Only rate-check LARGE deltas (> 50). Small deltas (≤ 50) are OCR noise on the same
    # counter digit — e.g. reading 18917 then 18929 in 0.5s is noise, not a real jump.
    # The rate check exists to catch cross-counter jumps like 388→2700, not digit noise.
    ocr_noise_band = 50
    if elapsed_seconds > 0 and not is_reset and not coming_from_reset and delta > ocr_noise_band:
        actual_rate = delta / elapsed_seconds
        rate_ok = delta <= max_allowed

        if not rate_ok:
            return ValidationResult(
                is_valid=False,
                confidence_ok=True,
                rate_ok=False,
                direction_ok=True,
                reason=(
                    f"Counter jumped {delta} in {elapsed_seconds:.1f}s "
                    f"({actual_rate:.1f}/sec) — exceeds machine max "
                    f"({max_rate_per_second}/sec × 1.2 buffer = {max_allowed:.0f} max). "
                    f"Rejected — likely OCR misread."
                ),
            )
    else:
        rate_ok = True

    # ── All checks passed ─────────────────────────────────────────────
    actual_rate = delta / elapsed_seconds if elapsed_seconds > 0 else 0
    return ValidationResult(
        is_valid=True,
        confidence_ok=True,
        rate_ok=True,
        direction_ok=True,
        reason=(
            f"Valid. Counter: {prev_value} → {new_value} "
            f"(+{delta} in {elapsed_seconds:.1f}s, {actual_rate:.1f}/sec)."
        ),
    )
