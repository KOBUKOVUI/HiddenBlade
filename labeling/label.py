#clayzzz

import csv
import string
import math
from zxcvbn import zxcvbn
from tqdm import tqdm

# Read passwords from file and clean each line
def read_passwords(file_path):
    passwords = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as pass_file: #escape error in "utf-8"
        for line in pass_file:
            cleaned_line = line.strip().replace('\t', '').replace(' ', '') # strip empty line 
            if cleaned_line:
                passwords.append(cleaned_line)
    return passwords

# Calculate usng entropy 
def calculate_entropy(password):
    length = len(password)
    charset_size = 0

    # Estimate character set size based on content
    if any(c.islower() for c in password):
        charset_size += 26
    if any(c.isupper() for c in password):
        charset_size += 26
    if any(c.isdigit() for c in password):
        charset_size += 10
    if any(c in string.punctuation for c in password):
        charset_size += len(string.punctuation)
    if any(c.isspace() for c in password):
        charset_size += 1

    if charset_size == 0 or length == 0:
        return 0

    # Entropy = length * log2(character set size)
    entropy = length * math.log2(charset_size)
    return entropy

# Label by entropy score
def label_by_entropy(entropy):
    if entropy < 28:
        return 0, "weak"
    elif entropy < 36:
        return 1, "weak"
    elif entropy < 60:
        return 2, "medium"
    elif entropy < 70:
        return 3, "fair"
    else:
        return 4, "strong"

# Use zxcvbn to evaluate password (base on ["score"]), then refine label using entropy for "strong" and "fair" cases
def evalute_passwords_zxcvbn(password):
    # Skip passwords longer than 72 characters (zxcvbn limitation)
    if len(password) >= 72:
        return None, None

    # Get zxcvbn score and take the label
    result = zxcvbn(password)
    score = result['score']

    if score <= 1:
        label = "weak"
    elif score == 2:
        label = "medium"
    elif score == 3:
        label = "fair"
    else:
        label = "strong"

    # If zxcvbn defines strong/fair but entropy is low, downgrade it base on entropy label function
    if label in ["strong", "fair"]:
        entropy = calculate_entropy(password)
        if entropy <= 60:
            score, label = label_by_entropy(entropy)
    
    return score, label

# Write to csv
def write_labeled_passwords_to_csv(passwords, output_file):
    results = []
    for passwd in tqdm(passwords, desc="🔍 Labeling", unit="pwd", dynamic_ncols=True):
        score, label = evalute_passwords_zxcvbn(passwd)
        if score is not None:
            results.append([passwd, score, label])

    with open(output_file, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["password", "score", "label"]) 
        writer.writerows(results) # use writer to avoid comma "," 



def main():
    input_file = "..."  # Input file path - change this to your own file
    output_file = "..."  # Output path and csv name

    passwords = read_passwords(input_file)
    write_labeled_passwords_to_csv(passwords, output_file)

    print("\n" + "=" * 60)
    print("🔥🔥🔥  LABELING COMPLETED - CHECK YOUR CSV FILE!  🔥🔥🔥")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
