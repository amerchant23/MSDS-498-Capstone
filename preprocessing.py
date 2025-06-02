# preprocessing.py
def select_description(row):
    # Example logic — replace with yours
    return row['Description'] if isinstance(row['Description'], str) else ''
