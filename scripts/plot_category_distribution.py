import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import argparse

def plot_category_distribution(input_csv: str, output_image: str):
    """
    Reads a CSV file, counts category occurrences, and saves a bar plot.
    """
    if not os.path.exists(input_csv):
        print(f"Error: Input file not found at {input_csv}")
        return

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if 'categories' not in df.columns:
        print("Error: 'categories' column not found in the CSV file.")
        return

    # Handle cases where categories might be NaN/float
    df.dropna(subset=['categories'], inplace=True)
    
    # Split comma-separated categories and count them
    all_categories = [cat.strip() for sublist in df['categories'].dropna().str.split(',') for cat in sublist if cat.strip()]

    # Count the occurrences of each category
    category_counts = Counter(all_categories)

    # Convert the Counter to a pandas Series for easy plotting
    counts_series = pd.Series(category_counts).sort_values(ascending=False)

    if not category_counts:
        print("No categories found to plot.")
        return
        
    # Create a DataFrame for plotting
    counts_df = pd.DataFrame(category_counts.items(), columns=['Category', 'Count']).sort_values('Count', ascending=False)

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.barplot(x=counts_series.values, y=counts_series.index, palette="viridis")
    plt.xlabel("Anzahl der Bilder", fontsize=12)
    plt.ylabel("Kategorien", fontsize=12)
    
    # Add the total number of images to the title
    total_images = df.shape[0]
    plt.title(f"Verteilung der Bildkategorien (insgesamt {total_images} Bilder)", fontsize=16)
    
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(output_image)
        print(f"Successfully saved plot to {output_image}")
    except Exception as e:
        print(f"Error saving plot: {e}")

def main():
    parser = argparse.ArgumentParser(description="Erstellt ein Balkendiagramm der Kategorienverteilung aus einer CSV-Datei.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="reports/real_descriptions_categorized.csv",
        help="Pfad zur CSV-Datei mit den kategorisierten Bildbeschreibungen."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="reports/category_distribution.png",
        help="Pfad zur Speicherzieldatei für das Diagramm."
    )
    args = parser.parse_args()

    plot_category_distribution(args.csv_path, args.output_path)

if __name__ == "__main__":
    main()
