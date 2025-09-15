import csv
import os

def extract_unique_categories():
    input_file = 'categorys/categorized_overview.csv'
    output_file = 'categorys/unique_categories.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    unique_categories = set()
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            
            # Find the index of the 'Categories' column
            try:
                category_index = header.index('Categories')
            except ValueError:
                print("Error: 'Categories' column not found in the CSV header.")
                return

            for row in reader:
                if row and len(row) > category_index:
                    # Split categories by comma and strip whitespace
                    categories = [cat.strip() for cat in row[category_index].split(',')]
                    unique_categories.update(categories)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Remove any empty strings that might have resulted from parsing
    unique_categories.discard('')

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Category'])
            for category in sorted(list(unique_categories)):
                writer.writerow([category])
        print(f"Successfully created {output_file} with {len(unique_categories)} unique categories.")
    except Exception as e:
        print(f"An error occurred while writing the output file: {e}")


if __name__ == "__main__":
    extract_unique_categories()
