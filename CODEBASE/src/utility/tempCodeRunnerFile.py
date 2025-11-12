import pandas as pd
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG = {
    "PATH": {
        "DATASET_PATH": "../dataset",
        "TRAIN_DATASET_PATH": "../../dataset/train.csv",
        "TEST_DATASET_PATH": "../../dataset/test.csv",
        "TRAIN_CLEANED_PATH": "../../dataset/train_cleaned.csv",
        "TEST_CLEANED_PATH": "../../dataset/test_cleaned.csv",
    }
}

# LOADING DATA
train_data = pd.read_csv(CONFIG["PATH"]["TRAIN_DATASET_PATH"])
test_data = pd.read_csv(CONFIG["PATH"]["TEST_DATASET_PATH"])

# HELPER FUNCTION
def _segregator(data):
    # segregates the data into different strings:
    # csv entry -> [item_name, bullet_points, value, unit]
    # item_name = Item Name string,
    # value = Value float,
    # unit = Unit string,
    # bullet_points = list of Bullet Point strings
    # image_link = image URL string
    catalog_content = data['catalog_content']
    lines = catalog_content.split('\n')
    item_name = None
    bullet_points = []
    value = None
    unit = None

    unit_mapping = {
        # Weight units
        'Ounce': 'ounce', 'ounce': 'ounce', 'oz': 'ounce', 'Oz': 'ounce', 'OZ': 'ounce', 'ounces': 'ounce', 'Ounces': 'ounce',
        'Pound': 'pound', 'pound': 'pound', 'lb': 'pound', 'LB': 'pound', 'pounds': 'pound', 'Pounds': 'pound',
        'Gram': 'gram', 'gram': 'gram', 'gramm': 'gram', 'grams': 'gram', 'Grams(gm)': 'gram', 'gr': 'gram',
        'kg': 'kilogram',
        
        # Volume units
        'Fl Oz': 'fluid_ounce', 'fl oz': 'fluid_ounce', 'FL Oz': 'fluid_ounce', 'FL OZ': 'fluid_ounce', 'Fl. Oz': 'fluid_ounce', 
        'Fl oz': 'fluid_ounce', 'fl. oz.': 'fluid_ounce', 'Fluid Ounce': 'fluid_ounce', 'fluid ounce': 'fluid_ounce', 
        'fluid ounces': 'fluid_ounce', 'Fluid Ounces': 'fluid_ounce', 'fluid ounce(s)': 'fluid_ounce', 'Fl.oz': 'fluid_ounce', 
        'Fl Ounce': 'fluid_ounce', 'Fluid ounce': 'fluid_ounce',
        'Liters': 'liter', 'ltr': 'liter',
        'milliliter': 'milliliter', 'millilitre': 'milliliter', 'mililitro': 'milliliter',
        
        # Count/Quantity units
        'Count': 'count', 'count': 'count', 'ct': 'count', 'CT': 'count', 'COUNT': 'count',
        'each': 'each', 'Each': 'each',
        'pack': 'pack', 'Pack': 'pack', 'packs': 'pack', 'Packs': 'pack', 'PACK': 'pack',
        'bottle': 'bottle', 'Bottle': 'bottle', 'bottles': 'bottle',
        'can': 'can', 'Can': 'can',
        'bag': 'bag', 'Bag': 'bag',
        'box': 'box', 'Box': 'box',
        'pouch': 'pouch', 'Pouch': 'pouch',
        'jar': 'jar', 'Jar': 'jar',
        'carton': 'carton', 'Carton': 'carton',
        'bucket': 'bucket', 'Bucket': 'bucket',
        
        # Other/miscellaneous -> 'other'
        'Piece': 'other', 'K-Cups': 'other', 'Tea bags': 'other', 'Tea Bags': 'other', 'Ziplock bags': 'other',
        'Paper Cupcake Liners': 'other', 'capsule': 'other', 'unità': 'other', 'units': 'other',
        'Foot': 'other', 'Sq Ft': 'other', 'sq ft': 'other', 'in': 'other', 'per Carton': 'other', 'per Box': 'other',
        'Per Package': 'other', 'BOX/12': 'other', '7,2 oz': 'other', 'product_weight': 'other',
        
        # Ambiguous or rare units -> 'other'
        '---': 'other', '-': 'other', '1': 'other', '8': 'other', '24': 'other', 'varies': 'other', 'various': 'other', 'N/A': 'other', 'n/a': 'other', 'NA': 'other', 'na': 'other'
    }

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('Item Name:'):
            item_name = line.replace('Item Name:', '').strip()
        elif line.startswith('Bullet Point'):
            bp = line.split(':', 1)[1].strip() if ':' in line else line
            bullet_points.append(bp)
        elif line.startswith('Value:'):
            val_str = line.replace('Value:', '').strip()
            try:
                value = float(val_str)
            except ValueError:
                value = None
        elif line.startswith('Unit:'):
            unit = line.replace('Unit:', '').strip()
    
    # Normalize unit using mapping
    if unit:
        unit = unit_mapping.get(unit, unit)  # Default to original if not in mapping
    
    return {
        'sample_id': data['sample_id'],
        'item_name': item_name,
        'bullet_points': bullet_points,  # List
        'value': value,
        'unit': unit,
        'image_link': data['image_link'],
        'price': data.get('price', None)  # Only for train
    }

# CLEANING AND SAVING
def clean_and_save(data, output_path, is_train=True):
    cleaned_rows = []
    for _, row in data.iterrows():
        cleaned = _segregator(row)
        cleaned_rows.append(cleaned)
    
    cleaned_df = pd.DataFrame(cleaned_rows)
    # Convert bullet_points list to JSON string for CSV storage
    cleaned_df['bullet_points'] = cleaned_df['bullet_points'].apply(json.dumps)
    cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

# Clean and save train data
clean_and_save(train_data, CONFIG["PATH"]["TRAIN_CLEANED_PATH"], is_train=True)
# Clean and save test data (without price)
clean_and_save(test_data, CONFIG["PATH"]["TEST_CLEANED_PATH"], is_train=False)

# LOADING CLEANED DATA
train_cleaned = pd.read_csv(CONFIG["PATH"]["TRAIN_CLEANED_PATH"])
test_cleaned = pd.read_csv(CONFIG["PATH"]["TEST_CLEANED_PATH"])

# Dropping rows with missing critical values
train_cleaned = train_cleaned.dropna(subset=['value', 'unit', 'item_name'])
train_cleaned = train_cleaned.reset_index(drop=True)

# Remove outliers
train_cleaned = train_cleaned[(train_cleaned['value'] >= 0) & (train_cleaned['value'] <= 561.4)]
train_cleaned = train_cleaned[(train_cleaned['price'] >= 0) & (train_cleaned['price'] <= 437.578)]
train_cleaned = train_cleaned.reset_index(drop=True)

# Print the frequency of each value and price
print("Value Frequency Distribution:")
print(pd.cut(train_cleaned['value'], bins=10).value_counts().sort_index())
print("\nPrice Frequency Distribution:")
print(pd.cut(train_cleaned['price'], bins=10).value_counts().sort_index())

# Print the frequency of each value and price through plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(train_cleaned['value'].dropna(), bins=50, kde=True)
plt.title('Value Distribution')
plt.subplot(1, 2, 2)
sns.histplot(train_cleaned['price'].dropna(), bins=50, kde=True)
plt.title('Price Distribution')
plt.show()

# Store the cleaned data again
train_cleaned.to_csv(CONFIG["PATH"]["TRAIN_CLEANED_PATH"], index=False)

# Print unique units and their counts, not as a dataframe but as a list
print("Unique Units and their Counts:")
print(train_cleaned['unit'].value_counts().to_dict())