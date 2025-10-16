from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class IEQConfig:
    num_zones: int = 5
    co2_gen_rate_lph_per_person: float = 18.0
    co2_outdoor_ppm: float = 420.0
    co2_decay_per_hour: float = 0.35
    pm25_outdoor_ugm3: float = 12.0
    pm10_outdoor_ugm3: float = 20.0
    voc_random_walk_sigma: float = 2.0
    seed: int = 42


def simulate_ieq(index: pd.DatetimeIndex, weather: pd.DataFrame, occupancy: pd.DataFrame, hvac_df: pd.DataFrame, cfg: IEQConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    dt_hours = (index[1] - index[0]).total_seconds() / 3600.0
    num_zones = cfg.num_zones

    # CO2 (ppm) mass balance
    co2 = np.zeros((len(index), num_zones))
    co2[0, :] = cfg.co2_outdoor_ppm + 50

    # TVOC random walk
    tvoc = np.zeros((len(index), num_zones))
    tvoc[0, :] = 150.0 + rng.normal(0, 10, num_zones)

    # PM from outdoor + tiny internal + removal
    pm25 = np.zeros((len(index), num_zones))
    pm10 = np.zeros((len(index), num_zones))
    pm25[0, :] = cfg.pm25_outdoor_ugm3
    pm10[0, :] = cfg.pm10_outdoor_ugm3

    # Lux from solar + electric lighting proxy from damper/occupied
    lux = np.zeros((len(index), num_zones))

    # PIR motion proxy & dBA from occupancy
    motion = np.zeros((len(index), num_zones))
    dba = np.zeros((len(index), num_zones))

    out_rh = np.clip(weather["ambient_rh_pct"].values, 15, 100)
    solar = np.clip(weather["solar_irradiance_wm2"].values, 0, None)

    for t in range(1, len(index)):
        occ = occupancy[[c for c in occupancy.columns if c.startswith("zone_")]].iloc[t].values
        damper = hvac_df[[c for c in hvac_df.columns if c.endswith("vav_damper_pos")]].iloc[t].values
        damper = damper.reshape(1, -1)
        vent_factor = 0.3 + 0.7 * np.clip(damper, 0, 1)

        # CO2 update
        gen_ppm_per_h = (cfg.co2_gen_rate_lph_per_person * occ)  # proportional
        co2[t, :] = (
            co2[t-1, :] + dt_hours * (gen_ppm_per_h - cfg.co2_decay_per_hour * (co2[t-1, :] - cfg.co2_outdoor_ppm))
        )
        co2[t, :] = np.clip(co2[t, :], cfg.co2_outdoor_ppm, None)

        # TVOC random walk with mean reversion
        tvoc[t, :] = np.clip(0.98 * tvoc[t-1, :] + rng.normal(0, cfg.voc_random_walk_sigma, num_zones) + 0.05 * occ, 50, 2000)

        # PM dynamics
        pm25[t, :] = np.clip(0.92 * pm25[t-1, :] + 0.06 * cfg.pm25_outdoor_ugm3 + 0.02 * np.sqrt(occ), 4, 200)
        pm10[t, :] = np.clip(0.93 * pm10[t-1, :] + 0.05 * cfg.pm10_outdoor_ugm3 + 0.02 * np.sqrt(occ), 7, 300)

        # Lux: daylight + electric to a target if occupied
        daylight = 1200 * (solar[t] / 1000.0)
        electric = 0.0 + 600.0 * (occ > 5).astype(float)
        lux[t, :] = np.clip(daylight + electric, 5, 2500)

        # Motion: thresholded with noise (0/1 pulses)
        motion_prob = np.clip(occ / (occ.max() + 1e-6), 0, 1)
        motion[t, :] = (rng.random(num_zones) < (0.1 + 0.7 * motion_prob)).astype(float)

        # dBA: background + occupancy contribution + motion bursts
        dba[t, :] = np.clip(38 + 8 * np.log10(1 + occ) + 2.0 * motion[t, :], 35, 85)

    data = {}
    for z in range(num_zones):
        data[f"zone_{z+1}_co2_ppm"] = co2[:, z]
        data[f"zone_{z+1}_tvoc_ppb"] = tvoc[:, z]
        data[f"zone_{z+1}_pm25_ugm3"] = pm25[:, z]
        data[f"zone_{z+1}_pm10_ugm3"] = pm10[:, z]
        data[f"zone_{z+1}_lux"] = lux[:, z]
        data[f"zone_{z+1}_noise_dba"] = dba[:, z]
        data[f"zone_{z+1}_motion"] = motion[:, z]

    # Include ambient RH for reference
    data["ambient_rh_pct"] = out_rh

    return pd.DataFrame(data, index=index)
