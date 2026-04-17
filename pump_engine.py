"""
pump_engine.py
══════════════
Standalone pump simulation engine — extracts *all* calculation logic,
probabilities, segment planning, and PumpState from the original
pump_sensor_stream_databricks_BRONZE_UPDATED.py **without any PySpark
or Delta Lake dependencies**.

Every constant, formula, and code path is identical to the original;
only the PySpark write path is removed.
"""

import numpy as np
from datetime import datetime, timezone

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

N_PUMPS          = 5
CYCLES_PER_PUMP  = 2000
MAX_WEAR_CYCLES  = 5000

TARGET_FAIL_FRAC = 0.09
TARGET_LEAK_FRAC = 0.10

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS — identical to original
# ══════════════════════════════════════════════════════════════════════════════

NORMAL = {
    "vibration":        (0.10, 0.40),
    "motor_temp":       (40.0, 70.0),
    "current_per_flow": (0.80, 1.20),
    "flow_deviation":   (0.00, 0.05),
    "pressure_delta":   (0.50, 1.50),
}
WORST = {
    "vibration":        1.50,
    "motor_temp":      115.0,
    "current_per_flow": 2.80,
    "flow_deviation":   0.50,
    "pressure_delta":   3.00,
}
WEIGHTS = {
    "vibration":        0.30,
    "motor_temp":       0.25,
    "current_per_flow": 0.20,
    "flow_deviation":   0.15,
    "pressure_delta":   0.10,
}
FEATURES = list(NORMAL.keys())

CAUSE_FAILURE = ["bearing", "overheat", "cavitation"]
CAUSE_LEAKAGE = ["pipe_joint", "seal_failure"]

FAILURE_LABEL = {
    "bearing":    "Bearing failure",
    "overheat":   "Overheating",
    "cavitation": "Cavitation",
}
LEAKAGE_LABEL = {
    "pipe_joint":   "Pipe joint loosening",
    "seal_failure": "Seal failure",
}

OUTPUT_COLUMNS = [
    "pump_id", "timestamp", "cycle_number",
    "vibration", "motor_temp", "current_per_flow", "flow_deviation", "pressure_delta",
    "wear_ratio",
    "probability_of_failure_in_percentage", "probability_of_leakage_in_percentage",
    "leakage_flag", "leakage_reason",
    "failure_flag", "failure_reason",
]

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS — identical to original
# ══════════════════════════════════════════════════════════════════════════════

def sample_normal_operating(feature, rng, noise=0.07):
    lo, hi = NORMAL[feature]
    mid    = (lo + hi) / 2.0
    sigma  = (hi - lo) * noise
    return float(np.clip(rng.normal(mid, sigma), lo * 0.92, hi * 1.08))


INJECT_RAMP_TICKS = 100
INJECT_RECOVERY_TICKS = 17


def normalise_feature(feature, value):
    lo, _ = NORMAL[feature]
    worst = WORST[feature]
    return float(np.clip((value - lo) / (worst - lo), 0.0, 1.0))


def compute_health_score(feats):
    return float(np.clip(
        sum(WEIGHTS[f] * normalise_feature(f, feats[f]) for f in FEATURES),
        0.0, 1.0
    ))


def lerp(a, b, t):
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


def health_score_to_probs(sc, wear):
    p_f = float(np.clip(sc * 18 + wear * 12, 0.0, 28.0))
    p_l = float(np.clip(sc * 14 + wear * 10, 0.0, 25.0))
    return p_f, p_l


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENT PLANNER — identical to original
# ══════════════════════════════════════════════════════════════════════════════

def plan_segments(n_rows, fail_frac, leak_frac, rng):
    target_fail  = int(round(n_rows * fail_frac))
    target_leak  = int(round(n_rows * leak_frac))
    target_quiet = n_rows - target_fail - target_leak

    def carve(budget, min_ep=30, max_ep=55):
        eps = []
        rem = budget
        while rem > 0:
            if rem <= max_ep:
                eps.append(rem); break
            l = int(rng.integers(min_ep, max_ep + 1))
            eps.append(l); rem -= l
        return eps

    fail_lengths = carve(target_fail)
    leak_lengths = carve(target_leak)
    n_ep = len(fail_lengths) + len(leak_lengths)

    gaps = [0] * (n_ep + 1)
    for _ in range(target_quiet):
        gaps[int(rng.integers(0, n_ep + 1))] += 1

    episodes = (
        [("failure", str(rng.choice(CAUSE_FAILURE)), l) for l in fail_lengths] +
        [("leakage", str(rng.choice(CAUSE_LEAKAGE)), l) for l in leak_lengths]
    )
    rng.shuffle(episodes)

    segments = []
    for i, (etype, cause, length) in enumerate(episodes):
        if gaps[i] > 0:
            segments.append({"type": "quiet", "cause": None,  "length": gaps[i]})
        segments.append(    {"type": etype,   "cause": cause, "length": length})
    if gaps[-1] > 0:
        segments.append({"type": "quiet", "cause": None, "length": gaps[-1]})

    return segments


# ══════════════════════════════════════════════════════════════════════════════
# PEAK FEATURES — identical to original
# ══════════════════════════════════════════════════════════════════════════════

def get_peak_features(cause, rng):
    peak = {f: lerp(NORMAL[f][0], NORMAL[f][1], 0.5) for f in FEATURES}
    if cause == "bearing":
        peak["vibration"]        = float(rng.uniform(1.20, 1.45))
        peak["current_per_flow"] = float(rng.uniform(1.60, 2.20))
        peak["motor_temp"]       = float(rng.uniform(65.0, 85.0))
    elif cause == "overheat":
        peak["motor_temp"]       = float(rng.uniform(95.0, 112.0))
        peak["current_per_flow"] = float(rng.uniform(1.50,  2.00))
        peak["vibration"]        = float(rng.uniform(0.40,  0.70))
    elif cause == "cavitation":
        peak["flow_deviation"]   = float(rng.uniform(0.38,  0.48))
        peak["pressure_delta"]   = float(rng.uniform(0.10,  0.22))
        peak["vibration"]        = float(rng.uniform(0.80,  1.20))
        peak["current_per_flow"] = float(rng.uniform(1.40,  1.80))
    elif cause == "pipe_joint":
        peak["pressure_delta"]   = float(rng.uniform(2.40,  2.90))
        peak["flow_deviation"]   = float(rng.uniform(0.35,  0.48))
        peak["vibration"]        = float(rng.uniform(0.40,  0.70))
    elif cause == "seal_failure":
        peak["vibration"]        = float(rng.uniform(1.05,  1.40))
        peak["current_per_flow"] = float(rng.uniform(2.00,  2.60))
        peak["flow_deviation"]   = float(rng.uniform(0.20,  0.40))
    return peak


# ══════════════════════════════════════════════════════════════════════════════
# PUMP STATE — identical to original (PySpark-free)
# ══════════════════════════════════════════════════════════════════════════════

class PumpState:
    def __init__(self, pump_id):
        self.pump_id = pump_id
        self.rng     = np.random.default_rng(seed=pump_id * 113 + 7)
        self.wear    = 0.0
        self.tick    = 0
        self._inject = None          # injected scenario override

        self.segments = plan_segments(
            CYCLES_PER_PUMP, TARGET_FAIL_FRAC, TARGET_LEAK_FRAC, self.rng
        )

        self._seg_queue = []
        for seg in self.segments:
            for k in range(seg["length"]):
                self._seg_queue.append((seg, k))

        self._episode_cache  = {}
        self._ep_start_feats = {}
        idx = 0
        for seg in self.segments:
            if seg["type"] != "quiet":
                slen      = seg["length"]
                ramp_len  = max(8, int(slen * 0.60))
                event_idx = ramp_len
                recov_len = max(0, slen - ramp_len - 1)
                peak_f    = get_peak_features(seg["cause"], self.rng)
                rec_f     = {f: sample_normal_operating(f, self.rng) for f in FEATURES}
                self._episode_cache[idx] = (peak_f, rec_f, ramp_len, event_idx, recov_len)
            idx += seg["length"]

        self._cycle_nums = []
        cycle_id = 1
        for seg in self.segments:
            if seg["type"] == "quiet":
                for _ in range(seg["length"]):
                    self._cycle_nums.append(cycle_id)
                    cycle_id += 1
            else:
                for _ in range(seg["length"]):
                    self._cycle_nums.append(cycle_id)
                cycle_id += 1

        self.prev_feats = {f: sample_normal_operating(f, self.rng) for f in FEATURES}

    # ── Scenario injection ────────────────────────────────────────────
    def inject_scenario(self, scenario_type, cause):
        """
        Inject a failure/leakage episode with a 100-tick ramp
        using the exact same ramp→peak→recovery math as the original.
        """
        rng       = np.random.default_rng(seed=self.pump_id * 113 + 7 + (self.tick % 997))
        ramp_len  = INJECT_RAMP_TICKS
        recov_len = INJECT_RECOVERY_TICKS
        e_len     = ramp_len + 1 + recov_len
        peak_f    = get_peak_features(cause, rng)
        rec_f     = {f: sample_normal_operating(f, rng) for f in FEATURES}
        event_idx = ramp_len

        start_feats = {
            f: float(np.clip(self.prev_feats[f],
                             NORMAL[f][0] * 0.90, NORMAL[f][1] * 1.10))
            for f in FEATURES
        }

        self._inject = {
            "type":        scenario_type,
            "cause":       cause,
            "ramp_len":    ramp_len,
            "event_idx":   event_idx,
            "recov_len":   recov_len,
            "peak_f":      peak_f,
            "rec_f":       rec_f,
            "start_feats": start_feats,
            "k":           0,
            "total":       e_len,
        }

    def next_row(self, now):
        """Advance one tick; return a dict with exactly OUTPUT_COLUMNS."""
        # If an injected scenario is active, use that path
        if self._inject is not None:
            return self._next_row_injected(now)

        if self.tick >= len(self._seg_queue):
            self.__init__(self.pump_id)

        seg, k    = self._seg_queue[self.tick]
        stype     = seg["type"]
        cause     = seg["cause"]
        seg_start = self.tick - k
        cycle_num = self._cycle_nums[self.tick]

        # ── Quiet ─────────────────────────────────────────────────────
        if stype == "quiet":
            feats = {
                f: self.prev_feats[f] * 0.3 + sample_normal_operating(f, self.rng) * 0.7
                for f in FEATURES
            }
            feats = {
                f: float(np.clip(feats[f], NORMAL[f][0] * 0.90, NORMAL[f][1] * 1.10))
                for f in FEATURES
            }
            sc        = compute_health_score(feats)
            self.wear = float(np.clip(self.wear + sc / MAX_WEAR_CYCLES, 0.0, 1.0))
            p_f, p_l  = health_score_to_probs(sc, self.wear)
            failure_flag = 0; leakage_flag = 0
            failure_reason = ""; leakage_reason = ""

        # ── Degradation episode ───────────────────────────────────────
        else:
            peak_feats, recover_feats, ramp_len, event_idx, recov_len = \
                self._episode_cache[seg_start]

            if k == 0:
                self._ep_start_feats[seg_start] = {
                    f: float(np.clip(self.prev_feats[f],
                                     NORMAL[f][0] * 0.90, NORMAL[f][1] * 1.10))
                    for f in FEATURES
                }
            start_feats = self._ep_start_feats[seg_start]
            is_failure  = (stype == "failure")

            if k < ramp_len:
                t = (k + 1) / (ramp_len + 1)
                feats = {
                    f: lerp(start_feats[f], peak_feats[f], t)
                       + self.rng.normal(0, (NORMAL[f][1] - NORMAL[f][0]) * 0.025)
                    for f in FEATURES
                }
                feats = {
                    f: float(np.clip(feats[f], NORMAL[f][0] * 0.40, WORST[f]))
                    for f in FEATURES
                }
                sc        = compute_health_score(feats)
                self.wear = float(np.clip(self.wear + sc * 1.8 / MAX_WEAR_CYCLES, 0.0, 1.0))
                if is_failure:
                    p_f = float(np.clip(15 + t * 69 + sc * 10, 0, 84))
                    p_l = float(np.clip(sc * 12 + self.wear * 8, 0, 38))
                else:
                    p_l = float(np.clip(15 + t * 69 + sc * 10, 0, 84))
                    p_f = float(np.clip(sc * 12 + self.wear * 8, 0, 38))

            elif k == event_idx:
                feats = {
                    f: float(np.clip(peak_feats[f] + self.rng.normal(0, 0.008),
                                     NORMAL[f][0] * 0.40, WORST[f]))
                    for f in FEATURES
                }
                sc        = compute_health_score(feats)
                self.wear = float(np.clip(self.wear + sc * 3.0 / MAX_WEAR_CYCLES, 0.0, 1.0))
                p_f = 92.5 if is_failure else float(np.clip(sc * 15, 0, 40))
                p_l = 91.5 if not is_failure else float(np.clip(sc * 15, 0, 40))

            else:
                r     = k - event_idx
                t_rec = r / max(recov_len, 1)
                feats = {
                    f: lerp(peak_feats[f], recover_feats[f], t_rec)
                       + self.rng.normal(0, (NORMAL[f][1] - NORMAL[f][0]) * 0.02)
                    for f in FEATURES
                }
                feats = {
                    f: float(np.clip(feats[f], NORMAL[f][0] * 0.70, NORMAL[f][1] * 1.10))
                    for f in FEATURES
                }
                sc        = compute_health_score(feats)
                self.wear = float(np.clip(self.wear + sc * 0.6 / MAX_WEAR_CYCLES, 0.0, 1.0))
                p_f, p_l  = health_score_to_probs(sc, self.wear)

            failure_flag   = 1 if is_failure else 0
            leakage_flag   = 0 if is_failure else 1
            failure_reason = FAILURE_LABEL.get(cause, "") if is_failure else ""
            leakage_reason = "" if is_failure else LEAKAGE_LABEL.get(cause, "")

        self.prev_feats = feats
        self.tick      += 1

        return {
            "pump_id":      self.pump_id,
            "timestamp":    now.strftime("%Y-%m-%d %H:%M:%S"),
            "cycle_number": cycle_num,
            "vibration":                            round(feats["vibration"],       4),
            "motor_temp":                           round(feats["motor_temp"],       4),
            "current_per_flow":                     round(feats["current_per_flow"], 4),
            "flow_deviation":                       round(feats["flow_deviation"],   4),
            "pressure_delta":                       round(feats["pressure_delta"],   4),
            "wear_ratio":                           round(self.wear,                 5),
            "probability_of_failure_in_percentage": round(float(np.clip(p_f, 0, 100)), 2),
            "probability_of_leakage_in_percentage": round(float(np.clip(p_l, 0, 100)), 2),
            "leakage_flag":    leakage_flag,
            "leakage_reason":  leakage_reason,
            "failure_flag":    failure_flag,
            "failure_reason":  failure_reason,
        }

    def _next_row_injected(self, now):
        """Process one tick of an injected scenario (same math as episode path)."""
        inj = self._inject
        k           = inj["k"]
        cause       = inj["cause"]
        ramp_len    = inj["ramp_len"]
        event_idx   = inj["event_idx"]
        recov_len   = inj["recov_len"]
        peak_feats  = inj["peak_f"]
        recover_feats = inj["rec_f"]
        start_feats = inj["start_feats"]
        is_failure  = (inj["type"] == "failure")

        # Ramp
        if k < ramp_len:
            t = (k + 1) / (ramp_len + 1)
            feats = {
                f: lerp(start_feats[f], peak_feats[f], t)
                   + self.rng.normal(0, (NORMAL[f][1] - NORMAL[f][0]) * 0.025)
                for f in FEATURES
            }
            feats = {
                f: float(np.clip(feats[f], NORMAL[f][0] * 0.40, WORST[f]))
                for f in FEATURES
            }
            sc        = compute_health_score(feats)
            self.wear = float(np.clip(self.wear + sc * 1.8 / MAX_WEAR_CYCLES, 0.0, 1.0))
            if is_failure:
                p_f = float(np.clip(15 + t * 69 + sc * 10, 0, 84))
                p_l = float(np.clip(sc * 12 + self.wear * 8, 0, 38))
            else:
                p_l = float(np.clip(15 + t * 69 + sc * 10, 0, 84))
                p_f = float(np.clip(sc * 12 + self.wear * 8, 0, 38))

        # Peak event
        elif k == event_idx:
            feats = {
                f: float(np.clip(peak_feats[f] + self.rng.normal(0, 0.008),
                                 NORMAL[f][0] * 0.40, WORST[f]))
                for f in FEATURES
            }
            sc        = compute_health_score(feats)
            self.wear = float(np.clip(self.wear + sc * 3.0 / MAX_WEAR_CYCLES, 0.0, 1.0))
            p_f = 92.5 if is_failure else float(np.clip(sc * 15, 0, 40))
            p_l = 91.5 if not is_failure else float(np.clip(sc * 15, 0, 40))

        # Recovery
        else:
            r     = k - event_idx
            t_rec = r / max(recov_len, 1)
            feats = {
                f: lerp(peak_feats[f], recover_feats[f], t_rec)
                   + self.rng.normal(0, (NORMAL[f][1] - NORMAL[f][0]) * 0.02)
                for f in FEATURES
            }
            feats = {
                f: float(np.clip(feats[f], NORMAL[f][0] * 0.70, NORMAL[f][1] * 1.10))
                for f in FEATURES
            }
            sc        = compute_health_score(feats)
            self.wear = float(np.clip(self.wear + sc * 0.6 / MAX_WEAR_CYCLES, 0.0, 1.0))
            p_f, p_l  = health_score_to_probs(sc, self.wear)

        failure_flag   = 1 if is_failure else 0
        leakage_flag   = 0 if is_failure else 1
        failure_reason = FAILURE_LABEL.get(cause, "") if is_failure else ""
        leakage_reason = "" if is_failure else LEAKAGE_LABEL.get(cause, "")

        self.prev_feats = feats
        self.tick      += 1
        inj["k"]       += 1

        # Episode finished → clear injection
        if inj["k"] >= inj["total"]:
            self._inject = None

        cycle_num = self._cycle_nums[min(self.tick - 1, len(self._cycle_nums) - 1)]

        return {
            "pump_id":      self.pump_id,
            "timestamp":    now.strftime("%Y-%m-%d %H:%M:%S"),
            "cycle_number": cycle_num,
            "vibration":                            round(feats["vibration"],       4),
            "motor_temp":                           round(feats["motor_temp"],       4),
            "current_per_flow":                     round(feats["current_per_flow"], 4),
            "flow_deviation":                       round(feats["flow_deviation"],   4),
            "pressure_delta":                       round(feats["pressure_delta"],   4),
            "wear_ratio":                           round(self.wear,                 5),
            "probability_of_failure_in_percentage": round(float(np.clip(p_f, 0, 100)), 2),
            "probability_of_leakage_in_percentage": round(float(np.clip(p_l, 0, 100)), 2),
            "leakage_flag":    leakage_flag,
            "leakage_reason":  leakage_reason,
            "failure_flag":    failure_flag,
            "failure_reason":  failure_reason,
        }

    @property
    def current_type(self):
        """Return the current segment type for status display."""
        if self._inject is not None:
            return self._inject["type"]
        if self.tick < len(self._seg_queue):
            seg, _ = self._seg_queue[self.tick]
            return seg["type"]
        return "quiet"

    @property
    def current_cause(self):
        if self._inject is not None:
            return self._inject["cause"]
        if self.tick < len(self._seg_queue):
            seg, _ = self._seg_queue[self.tick]
            return seg["cause"]
        return None


def get_utc_timestamp_floored():
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=0)
