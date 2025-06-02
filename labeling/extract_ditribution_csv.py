#clayzzz

import pandas as pd
from tqdm import tqdm

def analyze_label_distribution(file_path, output_report):
    # read only label column 
    df = pd.read_csv(file_path, usecols=["label"])
    
    #total number of passwords
    total = len(df)

    #Progress bar
    label_counts = {}
    tqdm.pandas(desc="🔍 Counting labels")
    
    #make a dictionary for types of labels (weak, medium, fair, strong)
    for label in tqdm(df["label"], desc="🔁 Processing", unit="pw", dynamic_ncols=True):
        label_counts[label] = label_counts.get(label, 0) + 1 # lay so dem hien tai, roi + 1, neu chua xuat hien thi cho la 0 roi + 1

    # Percentage calculation
    label_series = pd.Series(label_counts).sort_values(ascending=False) #transform a dictionary to a serie (pandas type) and sort it up to down
    label_percentages = (label_series / total) * 100

    
    # make a list contains lines in csv to put in file
    lines = []
    
    
    # CLI interface
    lines.append("=" * 50)
    lines.append("📊 Password Strength Label Distribution")
    lines.append("=" * 50)
    lines.append(f"{'Label':<10} {'Count':>15} {'Percentage':>15}")
    lines.append("-" * 50)
    
    # loop through the label_series (number of passwords each series was calculated before)
    for label in label_series.index:
        count = label_series[label] # take value at index [label]
        percent = label_percentages[label]
        lines.append(f"{label:<10} {count:>15,} {percent:>14.2f}%")
    
    #CLI interface
    lines.append("=" * 50)
    lines.append(f"🔢 Total passwords: {total:,}")
    lines.append("=" * 50)	

    # write to txt 
    print("\n".join(lines))
    with open(output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    input_csv = "..." # path to your labeled_data (csv format)
    output_txt = "..." # output path and file name
    analyze_label_distribution(input_csv, output_txt)
