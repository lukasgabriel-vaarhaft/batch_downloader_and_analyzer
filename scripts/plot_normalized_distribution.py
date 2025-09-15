import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import Counter

def plot_normalized_distribution(real_csv_path, gen_csv_path, output_path):
    """
    Generates a normalized, grouped bar chart for category distributions
    from REAL and GEN datasets, with percentages annotated on the bars.
    """
    data_frames = {}
    totals = {}
    
    # Load and process data for both sources
    for source, path in [('REAL', real_csv_path), ('GEN', gen_csv_path)]:
        try:
            df = pd.read_csv(path)
            all_categories = [cat.strip() for sublist in df['categories'].dropna().str.split(',') for cat in sublist if cat.strip()]
            counts = Counter(all_categories)
            total_images = len(df)
            
            # Calculate percentage
            percentages = {cat: (count / total_images) * 100 for cat, count in counts.items()}
            
            df_counts = pd.DataFrame(percentages.items(), columns=['Category', 'Percentage']).assign(Source=source)
            data_frames[source] = df_counts
            totals[source] = total_images
        except FileNotFoundError:
            print(f"Warning: {source} CSV not found at {path}. Skipping.")
            data_frames[source] = pd.DataFrame(columns=['Category', 'Percentage', 'Source'])
            totals[source] = 0

    # Combine data and prepare for plotting
    combined_df = pd.concat(data_frames.values())
    
    # Get top 20 categories based on REAL percentages for relevance
    if 'REAL' in data_frames:
        top_categories = data_frames['REAL'].nlargest(20, 'Percentage')['Category'].tolist()
        plot_df = combined_df[combined_df['Category'].isin(top_categories)]
    else:
        plot_df = combined_df # Fallback if REAL data is missing

    # Plotting
    plt.figure(figsize=(14, 10))
    ax = sns.barplot(data=plot_df, x='Percentage', y='Category', hue='Source', palette={'REAL': 'skyblue', 'GEN': 'salmon'})

    # Add annotations
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.1, p.get_y() + p.get_height() / 2,
                f'{width:.1f}%',
                va='center')

    plt.title(f'Normalisierter Vergleich der Kategorienverteilung\n(REAL: {totals["REAL"]} Bilder, GEN: {totals["GEN"]} Bilder)', fontsize=16)
    plt.xlabel("Prozentualer Anteil der Bilder (%)", fontsize=12)
    plt.ylabel("Top 20 Kategorien (basierend auf REAL)", fontsize=12)
    plt.legend(title='Datenquelle')
    plt.tight_layout()
    plt.xlim(right=ax.get_xlim()[1] * 1.15) # Adjust x-axis limit for annotations

    # Save the plot
    plt.savefig(output_path)
    print(f"Successfully saved normalized plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Erstellt ein normalisiertes, kombiniertes Balkendiagramm der Kategorienverteilung.")
    parser.add_argument("--real-csv", type=str, default="/Users/lukasgabriel/Documents/Projects/DataLabeling/real_descriptions_categorized.csv", help="Pfad zur REAL CSV-Datei.")
    parser.add_argument("--gen-csv", type=str, default="reports/gen_descriptions_categorized.csv", help="Pfad zur GEN CSV-Datei.")
    parser.add_argument("--output-path", type=str, default="reports/normalized_category_distribution.png", help="Pfad zur Speicherzieldatei für das Diagramm.")
    args = parser.parse_args()

    plot_normalized_distribution(args.real_csv, args.gen_csv, args.output_path)

if __name__ == "__main__":
    main()

