import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, confusion_matrix


from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


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
    Build the 24-hour energy and occupancy profile of each user.

    Parameters
    ----------
    user_hourly : pandas.DataFrame
        Output of aggregate_hourly(hourly, "user_id", ...), with columns
        user_id, hour, energy_kwh, occupied_h.

    Returns
    -------
    pandas.DataFrame
        One row per user, with 48 columns:
        - e_h00 to e_h23: share of the user's total energy consumption
          occurring in each hour of the day.
        - o_h00 to o_h23: share of the user's total occupied time
          occurring in each hour of the day.

        Each block sums to 1 per user (when the corresponding total is
        greater than 0). The profiles show the distribution of a user's
        energy consumption and occupied time across the day, independently
        of their total energy consumption or occupied time.
    """
    df = user_hourly.copy()
    df["hod"] = df["hour"].dt.hour

    blocks = []

    for col, prefix in [("energy_kwh", "e_h"), ("occupied_h", "o_h")]:
        p = df.pivot_table(
            index="user_id",
            columns="hod",
            values=col,
            aggfunc="sum",
            fill_value=0,
            observed=True
        )

        p = p.reindex(columns=range(24), fill_value=0)
        p = p.div(p.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        p.columns = [f"{prefix}{h:02d}" for h in p.columns]

        blocks.append(p)

    return pd.concat(blocks, axis=1)


 
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
        Extra session-level columns to aggregate per user (named aggregation):
        {"new_name": ("ev_column", agg)}. They go in the 'extra' group.
 
    Returns
    -------
    users : pandas.DataFrame   (one row per user, indexed by user_id)
    groups : dict   group name -> list of column names:
        - 'time':       share of the user's energy in each of 6 periods (sums to 1)
        - 'idle_ratio': idle hours / occupied hours (hoarding)
        - 'size':       log mean kWh per session
        - 'loyalty':    top_station_share, log_n_stations
        - 'regularity': log sessions per active week, log median gap (days)
        - 'weekend':    share of energy at weekends (= sum of we_* shares)
        - 'volume':     size variables (kept for backward compatibility)
        - 'extra':      columns from extra_aggs
        - 'profile':    raw-scale, interpretable variables used ONLY to
                        describe clusters (never as clustering features).
                        'peak_hour' is added later in build_feature_sets().
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
    # (kept in the table for profiling; the features use only idle_ratio)
    occ_total = uh.groupby("user_id", observed=True)["occupied_h"].sum().replace(0, np.nan)
    idle = uh.pivot_table(index="user_id", columns="period", values="idle_h",
                          aggfunc="sum", fill_value=0, observed=True)
    idle = idle.reindex(columns=periods, fill_value=0)
    idle = idle.div(occ_total, axis=0).fillna(0).add_prefix("idle_")
 
    users = energy.join(idle)
    users["idle_ratio"] = idle.sum(axis=1)
    users["weekend_share"] = users[["e_we_night", "e_we_day", "e_we_eve"]].sum(axis=1)
 
    # volume variables from the hourly table
    users["total_kwh"] = uh.groupby("user_id", observed=True)["energy_kwh"].sum()
    users["total_occupied_h"] = occ_total
    users["total_idle_h"] = uh.groupby("user_id", observed=True)["idle_h"].sum()
 
    # ---------- session-level features from ev ----------
    e = ev.dropna(subset=["user_id"]).copy()
    e["week"] = e["start"].dt.to_period("W")
    e = e.sort_values(["user_id", "start"])          # needed for gaps
    g = e.groupby("user_id", observed=True)
 
    mean_kwh = g["energy_kwh"].mean()
    mean_dur = g["total_hours"].mean()
 
    sess = pd.DataFrame({
        "n_sessions": g.size(),
        "n_stations": g["station"].nunique(),
        # raw versions (profiling) ...
        "mean_kwh_session": mean_kwh,
        "mean_dur_h": mean_dur,
        # ... and log versions (clustering features)
        "log_mean_kwh": np.log1p(mean_kwh),
        "log_mean_dur": np.log1p(mean_dur),
    })
    sess["active_weeks"] = g["week"].nunique()
    sess["sessions_per_active_week"] = sess["n_sessions"] / sess["active_weeks"]
    sess["log_n_stations"] = np.log1p(sess["n_stations"])
    sess["log_sessions_per_active_week"] = np.log1p(sess["sessions_per_active_week"])
 
    # regularity: median days between consecutive sessions (NaN if 1 session)
    gap_days = g["start"].diff().dt.total_seconds() / 86400
    median_gap = gap_days.groupby(e["user_id"]).median()
    sess["median_gap_days"] = median_gap                     # raw (profiling)
    sess["log_median_gap_days"] = np.log1p(median_gap)       # log (clustering)
 
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
 
    # revenue per kWh (only if the fee was aggregated into total_fee)
    profile_cols = ["mean_kwh_session", "mean_dur_h", "median_gap_days",
                    "total_kwh", "n_sessions", "active_weeks",
                    "sessions_per_active_week", "total_idle_h", "n_stations"]
    if "total_fee" in users.columns:
        users["fee_per_kwh"] = users["total_fee"] / users["total_kwh"].replace(0, np.nan)
        profile_cols.append("fee_per_kwh")
 
    groups = {
        "time": list(energy.columns),
        "idle_ratio": ["idle_ratio"],
        "size": ["log_mean_kwh"],
        "loyalty": ["top_station_share", "log_n_stations"],
        "regularity": ["log_sessions_per_active_week", "log_median_gap_days"],
        "weekend": ["weekend_share"],
        "volume": ["total_kwh", "total_occupied_h", "total_idle_h",
                   "n_sessions", "active_weeks", "n_stations",
                   "sessions_per_active_week", "log_mean_dur"],
        "extra": extra_cols,
        "profile": profile_cols,      # 'peak_hour' appended in build_feature_sets
    }
    return users, groups
 
 
def select_features(users, groups, *names):
    """
    Return the columns of the chosen groups, e.g.
    select_features(users, groups, "time", "idle_ratio")  -> tier B
    """
    cols = [c for n in names for c in groups[n]]
    return users[cols]
 
 
def build_feature_sets(user_hourly, ev, min_sessions=2, extra_aggs=None):
    """
    Returns a dict with the user-level tables (same index = kept users):
      'A'       : 24h energy share + 24h occupancy share + weekend_share (49 cols)
                  -> scale/PCA afterwards
      'B'       : time shares + idle_ratio + size + loyalty + regularity
                  (clustering features)
      'explore' : everything (raw profiling variables, idle_*, extra_aggs,
                  24h profiles, peak_hour) for exploratory analysis and for
                  profiling clusters. NOT used as clustering input.
      'groups'  : the groups dict
    """
    users, groups = build_user_table(user_hourly, ev, extra_aggs=extra_aggs)
    profile = build_user_profile(user_hourly)
 
    keep = users.index[users["n_sessions"] >= min_sessions]
    users = users.loc[keep].dropna(subset=groups["regularity"])
    profile = profile.loc[users.index]
 
    A = profile.join(users[groups["weekend"]])
    B = select_features(users, groups, "time", "idle_ratio", "size",
                        "loyalty", "regularity").copy()
    explore = users.join(profile)
 
    # peak hour of the day (needs the 24 hourly energy shares)
    e_cols = [f"e_h{h:02d}" for h in range(24)]
    explore["peak_hour"] = explore[e_cols].values.argmax(axis=1)
    if "peak_hour" not in groups["profile"]:
        groups["profile"].append("peak_hour")
 
    # safety check: profiling-only variables must not be clustering inputs
    assert set(groups["profile"]).isdisjoint(B.columns), \
        "A profiling variable leaked into B"
 
    print(f"Kept {len(users):,} users | "
          f"{users['total_kwh'].sum()/user_hourly['energy_kwh'].sum():.1%} of energy")
    return {"A": A, "B": B, "explore": explore, "groups": groups}
 

def compare_cluster_solutions(data, cluster1_col, cluster2_col):
    """
    Compare the clusters of two solutions using a confusion matrix.

    Args:
        data (pandas.DataFrame): DataFrame containing cluster assignments for both solutions.
        cluster1_col (str): Column name for the first solution's cluster assignments.
        cluster2_col (str): Column name for the second solution's cluster assignments.

    Returns:
        pandas.DataFrame: Confusion matrix comparing the clusters of the two solutions.
    """
    # Get unique cluster counts
    num_clusters1 = data[cluster1_col].nunique()
    num_clusters2 = data[cluster2_col].nunique()

    # Maximum number of clusters to account for
    max_clusters = max(num_clusters1, num_clusters2)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(data[cluster1_col], data[cluster2_col])

    # Create DataFrame with appropriate labels
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=[f'{cluster1_col} Cluster {i}' for i in range(max_clusters)],
        columns=[f'{cluster2_col} Cluster {i}' for i in range(max_clusters)]
    )

    # Return the relevant subset of the confusion matrix
    return conf_matrix_df.iloc[:num_clusters1, :num_clusters2]


def calculate_group_means(data, group_var, num_features=None):
    """
    Group the data by a specified variable and calculate the mean for each group.

    Args:
        data (pandas.DataFrame): The dataset.
        group_var (str): The variable to group by.
        num_features (int, optional): Number of features to include in the output. If None, include all features.

    Returns:
        pandas.DataFrame: Transposed DataFrame with mean values for each group.
    """
    # Group by the specified variable and compute mean for each group
    grouped_means = data.groupby(group_var).mean()

    # Select specified number of features and transpose the DataFrame
    if num_features is not None:
        grouped_means = grouped_means.iloc[:, :num_features]
    
    transposed_means = grouped_means.T

    # Return the transposed DataFrame
    return transposed_means