"""Per-neighborhood amenity-distance profiles — hybrid of three methods.

Method 1 — Fictional map: neighborhood centroids + school/hospital/transit hubs
on a seeded ~10×10 km plane; base distance = Euclidean nearest-hub distance.

Method 2 — Fixed neighborhood priors: documented multipliers for a few areas
with clear urban-form notes (historic core, campus edge, fringe parcels).
Remaining neighborhoods get a prior from modal MSZoning + median LotArea
density (smaller lots → denser fabric → slightly closer amenities).
Never uses SalePrice.

Method 3 — Listing-level Condition/Zoning enter later in listing_synthesis
(nudges + amenity_score 0–100).

Leakage discipline: SalePrice is never read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from faker import Faker

from src.config import RANDOM_SEED

_CITY_KM = 10.0
_N_SCHOOLS = 4
_N_HOSPITALS = 2
_N_TRANSIT = 5

# Method 2a — hand-crafted urban-form priors (NOT price ranks).
# Values multiply map distances (<1 closer, >1 farther).
_NEIGHBORHOOD_DISTANCE_PRIOR = {
    "OldTown": 0.92,   # historic grid / nearer civic fabric
    "SWISU": 0.90,     # campus-adjacent denser fabric
    "IDOTRR": 0.95,    # downtown-adjacent
    "Blueste": 0.93,   # compact RM pocket
    "BrDale": 0.94,    # compact townhouse fabric
    "ClearCr": 1.10,   # creek / larger parcels
    "Timber": 1.08,    # wooded fringe
    "Veenker": 1.06,   # golf / lower density edge
    "Gilbert": 1.04,   # newer peripheral subdivision pattern
    "NWAmes": 1.03,
}

# Method 2b — zoning → distance multiplier (denser zoning → closer services)
_ZONING_DISTANCE_PRIOR = {
    "RH": 0.85,
    "RM": 0.88,
    "FV": 0.90,
    "RL": 1.00,
    "RP": 1.00,
    "C (all)": 0.92,
    "C": 0.92,
    "A (agr)": 1.15,
    "A": 1.15,
    "I": 1.08,
    "I (all)": 1.08,
}


def _seeded_faker(seed: int) -> Faker:
    fake = Faker()
    fake.seed_instance(seed)
    return fake


def _draw_points(fake: Faker, n: int) -> np.ndarray:
    pts = np.empty((n, 2), dtype=float)
    for i in range(n):
        pts[i, 0] = fake.pyfloat(min_value=0.0, max_value=_CITY_KM)
        pts[i, 1] = fake.pyfloat(min_value=0.0, max_value=_CITY_KM)
    return pts


def _nearest_distance(origins: np.ndarray, hubs: np.ndarray) -> np.ndarray:
    d2 = ((origins[:, None, :] - hubs[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))


def build_amenity_hubs() -> pd.DataFrame:
    """Return the seeded fictional amenity hubs for audit/visualization."""
    fake = _seeded_faker(RANDOM_SEED + 1)
    rows = []
    for amenity, count in (
        ("school", _N_SCHOOLS),
        ("hospital", _N_HOSPITALS),
        ("transit", _N_TRANSIT),
    ):
        points = _draw_points(fake, count)
        for i, (x, y) in enumerate(points, start=1):
            rows.append({
                "amenity_type": amenity,
                "hub_id": f"{amenity}-{i}",
                "x_km": round(float(x), 3),
                "y_km": round(float(y), 3),
            })
    return pd.DataFrame(rows)


def _neighborhood_priors(ames: pd.DataFrame) -> pd.DataFrame:
    """Method 2: per-neighborhood distance multipliers (no SalePrice)."""
    modal_zone = (
        ames.groupby("Neighborhood")["MSZoning"]
        .agg(lambda s: s.mode().iloc[0])
    )
    lot_median = ames.groupby("Neighborhood")["LotArea"].median()
    # larger lots → higher rank → slightly farther amenities
    density_rank = lot_median.rank(method="average", pct=True)
    density_mult = 0.90 + 0.20 * density_rank  # ~0.90 (dense) … ~1.10 (sparse)

    rows = []
    for nbhd, zone in modal_zone.items():
        zone_m = _ZONING_DISTANCE_PRIOR.get(str(zone), 1.0)
        hand_m = _NEIGHBORHOOD_DISTANCE_PRIOR.get(nbhd, 1.0)
        dens_m = float(density_mult.loc[nbhd])
        # combine; clip so no neighborhood collapses to zero distance
        mult = float(np.clip(zone_m * dens_m * hand_m, 0.75, 1.35))
        rows.append({
            "Neighborhood": nbhd,
            "modal_mszoning": zone,
            "lotarea_median": float(lot_median.loc[nbhd]),
            "access_multiplier": round(mult, 3),
        })
    return pd.DataFrame(rows)


def build_neighborhood_profiles(
    ames: pd.DataFrame, hubs: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return one row per neighborhood: map bases × access prior.

    Only Neighborhood / MSZoning / LotArea are used from ``ames``.
    """
    rng = np.random.default_rng(RANDOM_SEED + 11)
    hubs = build_amenity_hubs() if hubs is None else hubs.copy()

    neighborhoods = (
        ames["Neighborhood"].drop_duplicates().sort_values().to_numpy()
    )
    n_nbhd = len(neighborhoods)
    centroids = rng.uniform(0.5, _CITY_KM - 0.5, size=(n_nbhd, 2))

    def _hub_points(amenity: str) -> np.ndarray:
        return hubs.loc[
            hubs["amenity_type"] == amenity, ["x_km", "y_km"]
        ].to_numpy(dtype=float)

    schools = _hub_points("school")
    hospitals = _hub_points("hospital")
    transit = _hub_points("transit")

    map_base = pd.DataFrame({
        "Neighborhood": neighborhoods,
        "nbhd_x_km": centroids[:, 0].round(3),
        "nbhd_y_km": centroids[:, 1].round(3),
        "map_dist_school_km": _nearest_distance(centroids, schools).round(3),
        "map_dist_hospital_km": _nearest_distance(centroids, hospitals).round(3),
        "map_dist_transit_km": _nearest_distance(centroids, transit).round(3),
    })

    priors = _neighborhood_priors(ames)
    profiles = map_base.merge(priors, on="Neighborhood", how="left")
    mult = profiles["access_multiplier"].to_numpy()

    # Method 1 × Method 2 → final neighborhood base used by listing synthesis
    profiles["base_dist_school_km"] = (
        profiles["map_dist_school_km"] * mult
    ).round(3)
    profiles["base_dist_hospital_km"] = (
        profiles["map_dist_hospital_km"] * mult
    ).round(3)
    profiles["base_dist_transit_km"] = (
        profiles["map_dist_transit_km"] * mult
    ).round(3)
    return profiles
