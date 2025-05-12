import pandas as pd

def resolve_ties_by_team(df):
    unique_times = df["time"].unique()
    resolved = []
    for i, t in enumerate(unique_times):
        candidates = df[df["time"] == t]
        if len(candidates) == 1:
            resolved.append(candidates.iloc[0])
        else:
            prev_team = resolved[-1]["Team"] if resolved else None
            next_team = None
            if i < len(unique_times) - 1:
                next_candidates = df[df["time"] == unique_times[i+1]]
                if not next_candidates.empty:
                    next_team = next_candidates.iloc[0]["Team"]

            chosen = None
            if prev_team and next_team:
                both = candidates[(candidates["Team"] == prev_team) & (candidates["Team"] == next_team)]
                if len(both) == 1:
                    chosen = both.iloc[0]
            if chosen is None and prev_team:
                match_prev = candidates[candidates["Team"] == prev_team]
                if len(match_prev) == 1:
                    chosen = match_prev.iloc[0]
            if chosen is None and next_team:
                match_next = candidates[candidates["Team"] == next_team]
                if len(match_next) == 1:
                    chosen = match_next.iloc[0]
            if chosen is None:
                chosen = candidates.iloc[0]
            resolved.append(chosen)
    return pd.DataFrame(resolved).reset_index(drop=True)
