"""
MetroGuard AI - Labeling Module
Constructs 2-hour early warning target labels and documents exclusion rules
for ongoing compressor failures and post-failure maintenance intervals.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


def assign_window_labels(
    timestamps: pd.Series,
    failure_events: List[Dict[str, Any]],
    horizon_hours: float = 2.0,
    ongoing_rule: str = "exclude",
    maintenance_hours: float = 6.0
) -> pd.DataFrame:
    """
    Constructs early warning binary target labels for time-indexed windows.
    
    Target definition:
        y_t = 1 if a reported compressor failure STARTS in (t, t + horizon_hours]
        y_t = 0 otherwise
        
    Window Status Categories:
        - "early_warning": t in [T_start - horizon, T_start), target = 1, included
        - "normal": normal operation, target = 0, included
        - "ongoing_failure": t in [T_start, T_end], target = NaN / excluded
        - "maintenance": t in (T_end, T_end + maintenance_hours], target = NaN / excluded
        
    Args:
        timestamps: Series of window end timestamps.
        failure_events: List of dicts with 'id', 'start', 'end' keys.
        horizon_hours: Early warning lookahead horizon (default 2.0 hours).
        ongoing_rule: "exclude" to remove active failure periods from early-warning modeling.
        maintenance_hours: Buffer hours post-failure to account for repair / stabling.
        
    Returns:
        DataFrame with columns ['timestamp', 'target', 'window_status', 'is_excluded', 'failure_id']
    """
    ts = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    n = len(ts)
    
    target = np.zeros(n, dtype=int)
    window_status = np.array(["normal"] * n, dtype=object)
    is_excluded = np.zeros(n, dtype=bool)
    failure_id = np.zeros(n, dtype=int)
    
    horizon_delta = pd.Timedelta(hours=horizon_hours)
    maint_delta = pd.Timedelta(hours=maintenance_hours)
    
    for event in failure_events:
        f_id = event["id"]
        t_start = pd.to_datetime(event["start"])
        t_end = pd.to_datetime(event["end"])
        
        # 1. Early warning window: t in [t_start - horizon, t_start)
        # For any t in this range, t_start falls strictly in (t, t + horizon]
        mask_early = (ts >= (t_start - horizon_delta)) & (ts < t_start)
        target[mask_early] = 1
        window_status[mask_early] = "early_warning"
        failure_id[mask_early] = f_id
        
        # 2. Ongoing failure window: t in [t_start, t_end]
        mask_ongoing = (ts >= t_start) & (ts <= t_end)
        if ongoing_rule == "exclude":
            window_status[mask_ongoing] = "ongoing_failure"
            is_excluded[mask_ongoing] = True
            failure_id[mask_ongoing] = f_id
            
        # 3. Post-failure maintenance: t in (t_end, t_end + maint_delta]
        mask_maint = (ts > t_end) & (ts <= (t_end + maint_delta))
        if maintenance_hours > 0:
            window_status[mask_maint] = "maintenance"
            is_excluded[mask_maint] = True
            failure_id[mask_maint] = f_id
            
    df_labels = pd.DataFrame({
        "timestamp": ts,
        "target": target,
        "window_status": window_status,
        "is_excluded": is_excluded,
        "failure_id": failure_id
    })
    
    return df_labels
