# ─────────────────────────────────────────────
#  src/pipeline/loader.py
#  Loads FastF1 sessions with disk cache.
# ─────────────────────────────────────────────

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict

import fastf1
import pandas as pd

from config import CACHE_DIR, YEAR, GRAND_PRIX, SESSIONS, DRIVER_CODES, TELEMETRY_CHANNELS

log = logging.getLogger(__name__)


def setup_cache() -> None:
    """Enable the fastf1 disk cache (speeds up repeat runs massively)."""
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    log.info("FastF1 cache enabled at '%s'", CACHE_DIR)


def load_session(year: int, gp: str, identifier: str) -> fastf1.core.Session:
    """
    Load and return a fully-loaded FastF1 session object.

    Parameters
    ----------
    year       : Championship year (e.g. 2025)
    gp         : Grand Prix name or round number (e.g. "British")
    identifier : Session type – "R" (Race), "Q" (Qualifying), "FP1" …
    """
    log.info("Loading %s %s %s …", year, gp, identifier)
    session = fastf1.get_session(year, gp, identifier)
    session.load(telemetry=True, weather=True, messages=False)
    log.info("Session loaded  –  %d laps available", len(session.laps))
    return session


def get_driver_laps(
    session: fastf1.core.Session,
    driver: str,
    drop_pit: bool = True,
) -> pd.DataFrame:
    """
    Return all laps for one driver, optionally dropping in / out-laps.

    Adds a float column ``LapTime_s`` (lap time in seconds) for convenience.
    """
    laps = session.laps.pick_drivers(driver)

    if drop_pit:
        laps = laps[~laps["PitOutTime"].notna() & ~laps["PitInTime"].notna()]

    laps = laps.copy()
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
    laps["Driver"] = driver
    return laps.reset_index(drop=True)


def get_best_lap(
    session: fastf1.core.Session,
    driver: str,
) -> fastf1.core.Lap:
    """Return the single fastest valid lap for a driver in this session."""
    laps = session.laps.pick_drivers(driver).pick_fastest()
    return laps


def get_telemetry_for_lap(lap: fastf1.core.Lap) -> pd.DataFrame:
    """
    Pull full car telemetry for a single lap and add distance-from-start.

    Returns a DataFrame with columns from TELEMETRY_CHANNELS where available,
    plus ``DRS_open`` (bool) derived from the raw DRS channel.
    """
    tel = lap.get_car_data().add_distance()

    # Merge positional data so we can map corners on the track map
    pos = lap.get_pos_data()
    tel = tel.merge_channels(pos)

    # Keep only the channels we defined in config (gracefully skip missing ones)
    available = [c for c in TELEMETRY_CHANNELS if c in tel.columns]
    tel = tel[available].copy()

    # Convenience boolean: DRS open when signal value > 8
    if "DRS" in tel.columns:
        tel["DRS_open"] = tel["DRS"] > 8

    return tel


# ── Convenience: load everything in one call ──────────────────────────────────

class SessionBundle:
    """
    Loads Race + Qualifying for the configured event and exposes
    per-driver laps and telemetry via simple attributes.

    Usage
    -----
    >>> bundle = SessionBundle()
    >>> bundle.load()
    >>> race_laps_ver = bundle.race_laps["VER"]
    >>> best_q_lap_nor = bundle.quali_best["NOR"]
    """

    def __init__(
        self,
        year: int = YEAR,
        gp: str = GRAND_PRIX,
        drivers: list[str] = DRIVER_CODES,
    ):
        self.year    = year
        self.gp      = gp
        self.drivers = drivers

        self.race_session:  fastf1.core.Session | None = None
        self.quali_session: fastf1.core.Session | None = None

        # Dict[driver_code -> DataFrame]
        self.race_laps:  Dict[str, pd.DataFrame] = {}
        self.quali_laps: Dict[str, pd.DataFrame] = {}

        # Dict[driver_code -> Lap]  (single fastest lap object)
        self.race_best:  Dict[str, fastf1.core.Lap] = {}
        self.quali_best: Dict[str, fastf1.core.Lap] = {}

    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> "SessionBundle":
        setup_cache()

        self.race_session  = load_session(self.year, self.gp, "R")
        self.quali_session = load_session(self.year, self.gp, "Q")

        for drv in self.drivers:
            self.race_laps[drv]  = get_driver_laps(self.race_session,  drv, drop_pit=True)
            self.quali_laps[drv] = get_driver_laps(self.quali_session, drv, drop_pit=False)

            self.race_best[drv]  = get_best_lap(self.race_session,  drv)
            self.quali_best[drv] = get_best_lap(self.quali_session, drv)

            log.info(
                "%s  –  race laps: %d  |  fastest race: %.3fs  |  fastest quali: %.3fs",
                drv,
                len(self.race_laps[drv]),
                self.race_best[drv]["LapTime"].total_seconds(),
                self.quali_best[drv]["LapTime"].total_seconds(),
            )

        return self

    # ──────────────────────────────────────────────────────────────────────────

    def get_telemetry(self, driver: str, session: str = "race") -> pd.DataFrame:
        """
        Return telemetry for the fastest lap of *driver* in the chosen session.

        Parameters
        ----------
        session : "race" | "quali"
        """
        lap = self.race_best[driver] if session == "race" else self.quali_best[driver]
        tel = get_telemetry_for_lap(lap)
        tel["Driver"]  = driver
        tel["Session"] = session
        return tel
