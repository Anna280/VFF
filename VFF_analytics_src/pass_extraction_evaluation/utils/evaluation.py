import pandas as pd

def count_pass_event_matches(df1, df2, tolerance=15):
    matches = 0
    for _, row1 in df1.iterrows():
        for _, row2 in df2.iterrows():
            if (abs(row1["Start Time [s]"] - row2["Start Time [s]"]) <= tolerance and
                abs(row1["End Time [s]"] - row2["End Time [s]"]) <= tolerance and
                row1["From"] == row2["From"] and
                row1["To"] == row2["To"]):
                matches += 1
                break

    total_df1 = len(df1)
    total_df2 = len(df2)
    recall = (matches / total_df1) * 100 if total_df1 > 0 else 0
    precision = (matches / total_df2) * 100 if total_df2 > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        "match_count": matches,
        "total_df1": total_df1,
        "total_df2": total_df2,
        "recall": recall,
        "precision": precision,
        "f1_score": f1_score
    }

def event_matches(pred_event, gt_event, tolerance):
    return (abs(pred_event["Start Time [s]"] - gt_event["Start Time [s]"]) <= tolerance and
            abs(pred_event["End Time [s]"] - gt_event["End Time [s]"]) <= tolerance and
            pred_event["From"] == gt_event["From"] and
            pred_event["To"] == gt_event["To"])

def evaluate_predictions_with_uncertainty(pred_df, gt_df, tolerance):
    TP = 0
    FP = 0
    matched_gt = set()
    correct_uncertainties = []
    incorrect_uncertainties = []

    for i, pred in pred_df.iterrows():
        match_found = False
        for j, gt in gt_df.iterrows():
            if event_matches(pred, gt, tolerance):
                match_found = True
                matched_gt.add(j)
                break
        if match_found:
            TP += 1
            correct_uncertainties.append(pred["uncertainty"])
        else:
            FP += 1
            incorrect_uncertainties.append(pred["uncertainty"])

    FN = len(gt_df) - len(matched_gt)
    confusion = {"TP": TP, "FP": FP, "FN": FN}
    return confusion, correct_uncertainties, incorrect_uncertainties
