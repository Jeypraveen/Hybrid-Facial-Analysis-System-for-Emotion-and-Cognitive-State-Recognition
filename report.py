import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

def parse_report(path):
    text = open(path, encoding='utf-8').read()
    # Derive session label from filename (or from inside the file)
    session = re.search(r"Video File:\s*(.+?)\.mp4", text)
    if session:
        label = session.group(1)
    else:
        label = path.split("/")[-1].rsplit(".", 1)[0]
    
    # Cognitive states
    att = float(re.search(r"Attentive\s*\│\s*[\d,]+\s*\│\s*([\d\.]+)%", text).group(1))
    dist = float(re.search(r"Distracted\s*\│\s*[\d,]+\s*\│\s*([\d\.]+)%", text).group(1))
    drow = float(re.search(r"Drowsy\s*\│\s*[\d,]+\s*\│\s*([\d\.]+)%", text).group(1))
    
    # Emotions
    emo = {}
    for name in ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]:
        pat = rf"{name}\s*│\s*[\d,]+\s*│\s*([\d\.]+)%"
        emo[name.lower()] = float(re.search(pat, text).group(1))
    
    # Performance
    fps = float(re.search(r"Average Processing Rate:\s*([\d\.]+)\s*FPS", text).group(1))
    total_frames = int(
        re.search(r"Total Frames Analyzed.*?:\s*([\d,]+)", text)
            .group(1)
            .replace(",", "")
    )
    return {
        "session": label,
        "attentive": att,
        "distracted": dist,
        "drowsy": drow,
        **emo,
        "fps": fps,
        "total_frames": total_frames
    }

def build_dataframe(folder_path):
    data = []
    file_list = sorted(glob.glob(f"{folder_path}/*.txt")) # Sort files to ensure consistent naming
    for i, filepath in enumerate(file_list):
        try:
            rec = parse_report(filepath)
            rec["session"] = f"s{i+1}" # Renaming to s1, s2, s3...
            data.append(rec)
        except Exception as e:
            print(f"Failed to parse {filepath}: {e}")
    df = pd.DataFrame(data)
    # No need to sort by session here as we are assigning them s1, s2...
    return df

def plot_graph_only(df):
    """Plot the graph without legend"""
    sessions = df["session"]
    keys = ["attentive", "distracted", "drowsy", "angry", "fear", "happy", "neutral", "sad", "surprise"]
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Set color cycle for bars using 'tab10' color map
    ax1.set_prop_cycle(cycler('color', plt.get_cmap('tab10').colors))
    
    bottom = pd.Series([0] * len(df))
    for k in keys:
        ax1.bar(sessions, df[k], bottom=bottom, label=k.capitalize())
        bottom += df[k]
        
    ax1.set_ylabel("Percentage (%)")
    ax1.set_xticklabels(sessions, rotation=90)
    
    # Add horizontal grid lines
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Secondary axis for fps and total_frames
    ax2 = ax1.twinx()
    ax2.plot(sessions, df["fps"], marker="o", linestyle="--", color="green", label="FPS")
    ax2.plot(sessions, df["total_frames"]/1000, marker="s", linestyle="-", color="red", label="Frames (×1000)")
    ax2.set_ylabel("FPS / Frames (×1000)")
    
    plt.title("Cognitive & Emotional States Across Sessions\nwith Processing Performance")
    plt.tight_layout()
    plt.show()

def plot_legend_only(df):
    """Plot only the legend in a separate figure"""
    keys = ["attentive", "distracted", "drowsy", "angry", "fear", "happy", "neutral", "sad", "surprise"]
    
    # Create figure for legend only
    fig = plt.figure(figsize=(14, 2))
    
    # Get the tab10 colors
    colors = plt.get_cmap('tab10').colors
    
    # Create all legend handles in one list
    handles = []
    
    # Create handles for bar chart legend
    for i, k in enumerate(keys):
        handle = plt.Rectangle((0,0),1,1, color=colors[i], label=k.capitalize())
        handles.append(handle)
        
    # Create handles for line plot legend
    line1 = plt.Line2D([0], [0], marker='o', linestyle='--', color='green', label='FPS')
    line2 = plt.Line2D([0], [0], marker='s', linestyle='-', color='red', label='Frames (×1000)')
    handles.extend([line1, line2])
    
    # Create single legend with all items
    legend = plt.figlegend(handles, [h.get_label() for h in handles], 
                          loc='center', bbox_to_anchor=(0.5, 0.5), 
                          ncol=len(handles), fancybox=True, shadow=True)
    
    # Remove axes
    plt.gca().set_axis_off()
    plt.tight_layout()
    plt.show()

def plot_summary_combined(df):
    """Plot both graph and legend together (original version)"""
    sessions = df["session"]
    keys = ["attentive", "distracted", "drowzy", "angry", "fear", "happy", "neutral", "sad", "surprise"]
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Set color cycle for bars using 'tab10' color map
    ax1.set_prop_cycle(cycler('color', plt.get_cmap('tab10').colors))
    
    bottom = pd.Series([0] * len(df))
    for k in keys:
        ax1.bar(sessions, df[k], bottom=bottom, label=k.capitalize())
        bottom += df[k]
        
    ax1.set_ylabel("Percentage (%)")
    ax1.set_xticklabels(sessions, rotation=90)
    
    # LEGEND HEIGHT CONTROL
    LEGEND1_HEIGHT = -0.22
    LEGEND2_HEIGHT = -0.27
    BOTTOM_MARGIN = 0.15
    
    # Bar chart legend
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, LEGEND1_HEIGHT), fancybox=True, shadow=True, ncol=len(keys))
    
    # Add horizontal grid lines
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Secondary axis for fps and total_frames
    ax2 = ax1.twinx()
    ax2.plot(sessions, df["fps"], marker="o", linestyle="--", color="green", label="FPS")
    ax2.plot(sessions, df["total_frames"]/1000, marker="s", linestyle="-", color="red", label="Frames (×1000)")
    ax2.set_ylabel("FPS / Frames (×1000)")
    
    # Line plot legend
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, LEGEND2_HEIGHT), fancybox=True, shadow=True, ncol=2)
    
    plt.title("Cognitive & Emotional States Across Sessions\nwith Processing Performance")
    plt.tight_layout(rect=[0, BOTTOM_MARGIN, 1, 1])
    plt.show()

if __name__ == "__main__":
    folder = "C:\\Projects\\FER\\REPORTS"  # Change this to your folder
    df = build_dataframe(folder)
    print(df)  # Optional: see the raw table

    # Choose one of these options:
    # Option 1: Graph only (no legend)
    print("Displaying graph without legend...")
    plot_graph_only(df)

    # Option 2: Legend only (separate figure)
    print("Displaying legend separately...")
    plot_legend_only(df)

    # Option 3: Combined graph with legend (original)
    # print("Displaying combined graph with legend...")
    # plot_summary_combined(df)