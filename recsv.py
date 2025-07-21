import os
import re
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# Path to the folder containing your .txt reports:
FOLDER_PATH = r'C:\Projects\FER\REPORTS'  # <-- set this once to your folder
# ────────────────────────────────────────────────────────────────────────────────

# The output CSV will be created inside FOLDER_PATH:
OUTPUT_CSV = os.path.join(FOLDER_PATH, 'reports_summary.csv')

# Regex patterns for each metric
patterns = {
    'video_file':         re.compile(r'Video File:\s*(.+)'),
    'session_duration':   re.compile(r'Session Duration:\s*([\d:]+)'),
    'total_frames':       re.compile(r'Total Frames Analyzed.*:\s*([\d,]+)'),
    'analysis_date':      re.compile(r'Analysis Date:\s*(.+)'),
    'raw_cog_detections': re.compile(r'Total Raw Cognitive State Detections:\s*([\d,]+)'),
    'raw_emo_detections': re.compile(r'Total Raw Emotion Detections:\s*([\d,]+)'),
    'fps':                re.compile(r'Average Processing Rate:\s*([\d.]+)\s*FPS'),
    'attentive_pct':      re.compile(r'Attentive\s*│\s*[\d,]+\s*│\s*([\d.]+)%'),
    'distracted_pct':     re.compile(r'Distracted\s*│\s*[\d,]+\s*│\s*([\d.]+)%'),
    'drowsy_pct':         re.compile(r'Drowsy\s*│\s*[\d,]+\s*│\s*([\d.]+)%'),
    'angry_pct':          re.compile(r'Angry\s*│\s*[\d,]+\s*│\s*([\d.]+)%'),
    'disgust_pct':        re.compile(r'Disgust\s*│\s*[\d,]+\s*│\s*([\d.]+)%'),
    'fear_pct':           re.compile(r'Fear\s*│\s*[\d,]+\s*│\s*([\d.]+)%'),
    'happy_pct':          re.compile(r'Happy\s*│\s*[\d,]+\s*│\s*([\d.]+)%'),
    'neutral_pct':        re.compile(r'Neutral\s*│\s*[\d,]+\s*│\s*([\d.]+)%'),
    'sad_pct':            re.compile(r'Sad\s*│\s*[\d,]+\s*│\s*([\d.]+)%'),
    'surprise_pct':       re.compile(r'Surprise\s*│\s*[\d,]+\s*│\s*([\d.]+)%'),
    'dominant_cog':       re.compile(r'Dominant Raw Cognitive State:\s*(\w+)'),
    'dominant_emo':       re.compile(r'Dominant Raw Emotion:\s*(\w+)')
}

def extract_metrics_from_text(text):
    """Return a dict of extracted values (or None) using our patterns."""
    row = {}
    for key, pat in patterns.items():
        m = pat.search(text)
        row[key] = m.group(1) if m else None
    return row

def main():
    records = []
    for fname in os.listdir(FOLDER_PATH):
        if not fname.lower().endswith('.txt'):
            continue
        with open(os.path.join(FOLDER_PATH, fname), 'r', encoding='utf-8') as f:
            txt = f.read()
        metrics = extract_metrics_from_text(txt)
        metrics['report_file'] = fname
        records.append(metrics)

    # Build DataFrame
    df = pd.DataFrame(records)

    # Clean numeric columns
    for col in ['total_frames','raw_cog_detections','raw_emo_detections']:
        if col in df:
            df[col] = (
                df[col]
                .str.replace(',', '', regex=False)
                .astype(float, errors='ignore')
            )

    pct_cols = [c for c in df.columns if c.endswith('_pct')] + ['fps']
    for col in pct_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Save CSV inside the same folder
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Aggregated {len(df)} reports into:\n   {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
