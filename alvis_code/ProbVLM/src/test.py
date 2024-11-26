import csv

def check_dataset(data_path):
    """
    Check if all rows in the dataset have exactly three elements.

    Args:
        data_path (str): Path to the dataset CSV file.
    """
    # Initialize counters
    valid_rows = 0
    invalid_rows = 0

    # Process dataset
    print(f"Checking dataset at: {data_path}")
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:  # Skip header row
                continue
            # Split by '|'
            if not row or len(row[0].split('|')) != 3:
                print(f"Invalid row at index {i}: {row}")
                invalid_rows += 1
            else:
                valid_rows += 1

    # Summary
    print(f"\nDataset Check Complete:")
    print(f"  Valid Rows: {valid_rows}")
    print(f"  Invalid Rows: {invalid_rows}")


if __name__ == "__main__":
    # Replace this path with your actual dataset path
    dataset_csv_path = "../datasets/flickr/results.csv"

    # Check the dataset
    check_dataset(dataset_csv_path)

