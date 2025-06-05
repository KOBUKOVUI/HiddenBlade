#clayzzz


import pandas as pd 
import math 
import string 
from tqdm import tqdm
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
    return round(entropy, 2)

# extract feature function
def extract_features(password, score): 
    # length of the password
    length = len(password)
    
    # char features
    count_lower = 0
    count_upper = 0
    count_digits = 0
    count_special_chars = 0 
    for c in password: 
        #lower case number
        if c.islower(): 
            count_lower = count_lower + 1
        #upper case number
        elif c.isupper(): 
            count_upper = count_upper + 1
        #digits number
        elif c.isdigit(): 
            count_digits = count_digits + 1
        #special char number
        elif c in string.punctuation:
            count_special_chars = count_special_chars + 1
            
    #entropy length
    entropy = calculate_entropy(password)
    
    #ratio
    ratio_lower = round(count_lower / length, 2)
    ratio_upper = round(count_upper / length, 2)
    ratio_digits = round(count_digits / length, 2)
    ratio_special_chars = round(count_special_chars / length, 2)
    
    #char_diveristy 
    char_diversity = 0 
    if count_lower > 0: 
        char_diversity = char_diversity + 1
    if count_digits > 0: 
        char_diversity = char_diversity + 1
    if count_upper > 0: 
        char_diversity = char_diversity + 1
    if count_special_chars > 0: 
        char_diversity = char_diversity + 1
    
    #common_phrase
    common_phrase_dict = [
    "password", "qwerty", "welcome", "iloveyou", "123", "abc", "fuckyou", 
    "admin", "1111", "letmein", "monkey", "dragon", "sunshine", "trustno1", 
    "qwertyuiop", "qwe", "princess", "master", "hello", "ashley", "1q2w3e4r", 
    "qazwsx", "superman", "123abc", "admin123", "test", "sunshine", "iloveu", 
    "1qaz2wsx", "1qazxsw2", "qwert", "starwars", "football", "shadow"
    ]

    password_lower = password.lower()
    
    common_phrase = 0
    for word in common_phrase_dict: 
        if word in password_lower: 
            common_phrase = 1
            break
            

    # dict to store features: 
    features = {
        'password': password,
        'score': score,
        'entropy': entropy,
        'length': length,
        'count_lower': count_lower,
        'ratio_lower': ratio_lower,
        'count_upper': count_upper,
        'ratio_upper': ratio_upper,
        'count_digits': count_digits,
        'ratio_digits': ratio_digits,
        'count_special_chars': count_special_chars,
        'ratio_special_chars': ratio_special_chars,
        'char_diversity': char_diversity,
        'common_phrase' : common_phrase,
    }
    
    return features
# working with files
def process_password_file(input_file, output_file): 
    
    #read csv file 
    df = pd.read_csv(input_file) 
    
    # array to hold features of passwords, array of dictionary records
    feature_list = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing passwords", dynamic_ncols=True): 
        # take the collumn passwd
        password = str(row['password'])
        # take the collumn score
        score = row['score'] # score was calculated by using zxcvbn before 
        
        #extract feture of current password
        feature = extract_features(password, score)
        
        feature_list.append(feature)
        
    #change to DataFrame format
    features_df = pd.DataFrame(feature_list)

    #write to new csv
    features_df.to_csv(output_file, index=False, quotechar='"')

input_file = "/home/hoang126/Desktop/workspace/csv/for_using/sanitized_labeled_data_for_extracting.csv" 
output_file = "/home/hoang126/Desktop/workspace/csv/for_using/6m4_extract_features.csv" 

process_password_file(input_file, output_file)
print("Succesfull extracting features")

