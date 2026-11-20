import numpy as np
import pandas as pd


# ----------------- Hourly Tables (one row per entity and hour) -------------------

def overlap_by_hour(df, end_col, overlap_col):
    """
    Expand each charging session into one row per clock hour it touches and
    compute how long the session overlaps each hour.

    Each session is treated as the half-open interval [start, end_col), so an
    end exactly on the hour does not create an extra zero-length row.
    Example: a session from 10:30 to 12:00 gives rows for 10:00 (0.5 h) and
    11:00 (1.0 h), but none for 12:00.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per charging session. Must contain a datetime column
        'start' and the datetime column named in `end_col`.
        Timestamps are naive Palo Alto local time and are not converted.
    end_col : str
        Column holding the end of the interval to analyse.
        - 'charge_end': start + charging_hours -> active charging time.
        - 'end': start + total_duration -> time the charger is occupied.
    overlap_col : str
        Name of the output column with the overlap in decimal hours
        (between 0 and 1), e.g. 'charging_h' or 'occupied_h'.

    Returns
    -------
    pandas.DataFrame
        One row per (session, clock hour), with:
        - all original columns of `df`;
        - 'session': integer id of the original session (position in `df`);
        - 'hour': start of the clock hour (e.g. 2019-05-03 10:00);
        - `overlap_col`: hours of overlap between the session and that hour.
        The index is reset (0..N-1), so it is unique.
    """
    df = df.reset_index(drop=True).copy()
    df["session"] = np.arange(len(df))

    # first clock hour touched by each session (10:35 -> 10:00)
    hour_start = df["start"].dt.floor("h")

    # number of clock hours touched; ceil avoids an extra row when the
    # end falls exactly on the hour
    n = np.ceil((df[end_col] - hour_start) / pd.Timedelta("1h")).astype(int)

    # repeat each session n times; reset_index keeps the index unique so
    # that the assignments below align row by row
    idx = np.repeat(np.arange(len(df)), n)
    out = df.iloc[idx].reset_index(drop=True)

    # 0, 1, 2, ... within each session -> beginning of each hourly interval
    out["k"] = out.groupby("session").cumcount()
    out["hour"] = hour_start.iloc[idx].reset_index(drop=True) + pd.to_timedelta(out["k"], unit="h")

    # overlap = [later of start and hour start, earlier of end and hour end]
    hour_end = out["hour"] + pd.Timedelta("1h")
    overlap_start = out["start"].where(out["start"] > out["hour"], out["hour"])
    overlap_end = out[end_col].where(out[end_col] < hour_end, hour_end)

    out[overlap_col] = (overlap_end - overlap_start).dt.total_seconds() / 3600
    return out.drop(columns="k")


def build_hourly_table(ev):
    """
    Transform charging sessions into a session-hour table with hourly energy,
    charging time, occupation time and idle time.

    Assumptions
    -----------
    - Charging starts at `start` (plug-in).
    - Active charging lasts `charging_hours`, at constant average power, so
      the session's energy is spread in proportion to the charging time that
      falls in each hour.
    - The vehicle stays connected, idle (no energy), until `end`.

    Parameters
    ----------
    ev : pandas.DataFrame
        Cleaned sessions (ev_clean.pkl) with columns: start, end,
        charging_hours, total_hours, energy_kwh, user_id, station.

    Returns
    -------
    pandas.DataFrame
        One row per (session, clock hour), with:
        - 'occupied_h': hours the port is occupied in that hour (0-1);
        - 'charging_h': hours of active charging in that hour (0-1);
        - 'idle_h': occupied_h - charging_h (plugged in but not charging);
        - 'energy_kwh': energy delivered in that hour (kWh);
        - 'session', 'hour', and the original session columns.
        Energy, occupied hours and charging hours sum back to the original totals.
    """
    ev = ev.reset_index(drop=True).copy()
    ev["charge_end"] = ev["start"] + pd.to_timedelta(ev["charging_hours"], unit="h")

    chg = overlap_by_hour(ev, end_col="charge_end", overlap_col="charging_h")
    occ = overlap_by_hour(ev, end_col="end", overlap_col="occupied_h")

    # the charging window lies inside the occupied window, so merge onto occ
    hourly = occ.merge(chg[["session", "hour", "charging_h"]],
                       on=["session", "hour"], how="left")
    hourly["charging_h"] = hourly["charging_h"].fillna(0)
    hourly["idle_h"] = hourly["occupied_h"] - hourly["charging_h"]

    # spread each session's energy over its charging hours
    hourly["energy_kwh"] = (hourly["energy_kwh"] * hourly["charging_h"]
                            / hourly["charging_hours"])

    # checks: nothing lost or created
    assert np.isclose(hourly["energy_kwh"].sum(), ev["energy_kwh"].sum())
    assert np.isclose(hourly["occupied_h"].sum(), ev["total_hours"].sum())
    assert np.isclose(hourly["charging_h"].sum(), ev["charging_hours"].sum())
    assert len(hourly) == len(occ)
    return hourly


def aggregate_hourly(hourly, by, cols):
    """
    Aggregate the session-hour table by an entity and clock hour.

    Parameters
    ----------
    hourly : pandas.DataFrame
        Output of build_hourly_table().
    by : str
        Entity column: 'user_id' (clustering) or 'station' (forecasting).
    cols : list of str
        Columns to sum, e.g. ['energy_kwh', 'charging_h', 'occupied_h', 'idle_h'].

    Returns
    -------
    pandas.DataFrame
        One row per (`by`, 'hour') with the summed columns. Rows with a
        missing `by` value (e.g. missing user_id) are excluded.
    """
    return (hourly.dropna(subset=[by])
                  .groupby([by, "hour"], observed=True)[cols]
                  .sum().reset_index())


# ----------------- Hourly Tables (one row per entity and hour) -------------------


def build_user_profile(user_hourly):
    """
    Build the hour-of-day energy profile of each user.

    Parameters
    ----------
    user_hourly : pandas.DataFrame
        Output of aggregate_hourly(hourly, "user_id", ...), with columns
        user_id, hour, energy_kwh.

    Returns
    -------
    pandas.DataFrame
        One row per user, one column per hour of day (0-23). Each value is
        the share of that user's total energy delivered in that hour, so
        every row sums to 1 and the profile shows when a user charges and
        not how much.
    """
    df = user_hourly.copy()
    df["hod"] = df["hour"].dt.hour
    profile = df.pivot_table(index="user_id", columns="hod",
                             values="energy_kwh", aggfunc="sum", fill_value=0)
    profile = profile.div(profile.sum(axis=1), axis=0)
    return profile.add_prefix("share_h")

def build_user_table(user_hourly, ev, extra_aggs=None):
    """
    Build one wide table with one row per user containing every candidate
    feature, plus a dict telling which columns belong to which group.

    Parameters
    ----------
    user_hourly : pandas.DataFrame
        Output of aggregate_hourly(hourly, "user_id", [...]) with columns
        user_id, hour, energy_kwh, charging_h, occupied_h, idle_h.
    ev : pandas.DataFrame
        Cleaned sessions (needs start, station, user_id, energy_kwh,
        total_hours).
    extra_aggs : dict, optional
        Extra session-level columns to aggregate per user, in pandas named
        aggregation style: {"new_name": ("ev_column", agg)}, where agg is a
        string or function. Example:
        {"mean_fee": ("fee", "mean"),
         "main_plug": ("plug_type", lambda s: s.mode().iloc[0])}
        These go in the 'extra' group (not used for clustering by default).

    Returns
    -------
    users : pandas.DataFrame
        One row per user, indexed by user_id.
    groups : dict
        group name -> list of column names. Groups:
        - 'time':       share of the user's energy in each period (sums to 1)
        - 'idle':       idle hours in each period / total occupied hours
        - 'idle_ratio': overall idle hours / occupied hours (equals the sum
                        of the 'idle' columns; use one OR the other)
        - 'behaviour':  session habits (size of charge, duration, loyalty, regularity)
        - 'volume':     size variables, for describing clusters only
        - 'extra':      columns from extra_aggs
    """
    periods = ["wd_night", "wd_day", "wd_eve", "we_night", "we_day", "we_eve"]

    # ---------- period features from the hourly table ----------
    uh = user_hourly.copy()
    hod = uh["hour"].dt.hour
    weekend = uh["hour"].dt.dayofweek >= 5
    night = (hod >= 22) | (hod < 6)
    day = (hod >= 6) & (hod < 17)

    uh["period"] = np.select(
        [~weekend & night, ~weekend & day, ~weekend,
          weekend & night,  weekend & day],
        ["wd_night", "wd_day", "wd_eve", "we_night", "we_day"],
        default="we_eve")

    # WHEN: share of the user's energy per period
    energy = uh.pivot_table(index="user_id", columns="period", values="energy_kwh",
                            aggfunc="sum", fill_value=0, observed=True)
    energy = energy.reindex(columns=periods, fill_value=0)
    energy = energy.div(energy.sum(axis=1), axis=0).fillna(0).add_prefix("e_")

    # HOW: idle hours per period relative to the user's total occupied hours
    occ_total = uh.groupby("user_id", observed=True)["occupied_h"].sum().replace(0, np.nan)
    idle = uh.pivot_table(index="user_id", columns="period", values="idle_h",
                          aggfunc="sum", fill_value=0, observed=True)
    idle = idle.reindex(columns=periods, fill_value=0)
    idle = idle.div(occ_total, axis=0).fillna(0).add_prefix("idle_")

    users = energy.join(idle)
    users["idle_ratio"] = idle.sum(axis=1)

    # volume variables from the hourly table
    users["total_kwh"] = uh.groupby("user_id", observed=True)["energy_kwh"].sum()
    users["total_occupied_h"] = occ_total
    users["total_idle_h"] = uh.groupby("user_id", observed=True)["idle_h"].sum()

    # ---------- session-level features from ev ----------
    e = ev.dropna(subset=["user_id"]).copy()
    e["week"] = e["start"].dt.to_period("W")
    g = e.groupby("user_id", observed=True)

    sess = pd.DataFrame({
        "n_sessions": g.size(),
        "n_stations": g["station"].nunique(),
        "log_mean_kwh": np.log1p(g["energy_kwh"].mean()),
        "log_mean_dur": np.log1p(g["total_hours"].mean()),
    })
    sess["active_weeks"] = g["week"].nunique()
    sess["sessions_per_active_week"] = sess["n_sessions"] / sess["active_weeks"]

    # loyalty: share of sessions at the user's most-used station
    counts = e.groupby(["user_id", "station"], observed=True).size()
    sess["top_station_share"] = (counts.groupby(level=0).max()
                                 / counts.groupby(level=0).sum())

    users = users.join(sess, how="left")

    # ---------- optional extra columns from ev ----------
    extra_cols = []
    if extra_aggs:
        extra = g.agg(**extra_aggs)
        users = users.join(extra, how="left")
        extra_cols = list(extra.columns)

    groups = {
        "time": list(energy.columns),
        "idle": list(idle.columns),
        "idle_ratio": ["idle_ratio"],
        "behaviour": ["log_mean_kwh", "log_mean_dur", "n_stations",
                      "top_station_share", "sessions_per_active_week"],
        "volume": ["total_kwh", "total_occupied_h", "total_idle_h",
                   "n_sessions", "active_weeks"],
        "extra": extra_cols,
    }
    return users, groups


def select_features(users, groups, *names):
    """
    Return the columns of the chosen groups, e.g.
    select_features(users, groups, "time", "idle")  -> tier B
    """
    cols = [c for n in names for c in groups[n]]
    return users[cols]