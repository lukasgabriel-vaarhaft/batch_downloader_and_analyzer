import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import Counter

def plot_combined_distribution(real_csv_path, gen_csv_path, output_path):
    """
    Generates a combined, grouped bar chart for category distributions
    from REAL and GEN datasets, with counts annotated on the bars.
    """
    # Load and process REAL data
    try:
        df_real = pd.read_csv(real_csv_path)
        all_categories_real = [cat.strip() for sublist in df_real['categories'].dropna().str.split(',') for cat in sublist if cat.strip()]
        counts_real = Counter(all_categories_real)
        df_counts_real = pd.DataFrame(counts_real.items(), columns=['Category', 'Count']).assign(Source='REAL')
        total_real_images = len(df_real)
    except FileNotFoundError:
        print(f"Warning: REAL CSV not found at {real_csv_path}. Skipping.")
        df_counts_real = pd.DataFrame(columns=['Category', 'Count', 'Source'])
        total_real_images = 0

    # Load and process GEN data
    try:
        df_gen = pd.read_csv(gen_csv_path)
        all_categories_gen = [cat.strip() for sublist in df_gen['categories'].dropna().str.split(',') for cat in sublist if cat.strip()]
        counts_gen = Counter(all_categories_gen)
        df_counts_gen = pd.DataFrame(counts_gen.items(), columns=['Category', 'Count']).assign(Source='GEN')
        total_gen_images = len(df_gen)
    except FileNotFoundError:
        print(f"Warning: GEN CSV not found at {gen_csv_path}. Skipping.")
        df_counts_gen = pd.DataFrame(columns=['Category', 'Count', 'Source'])
        total_gen_images = 0

    # Combine data and prepare for plotting
    combined_df = pd.concat([df_counts_real, df_counts_gen])
    
    # Get top 20 categories based on REAL counts for relevance
    top_categories = df_counts_real.nlargest(20, 'Count')['Category'].tolist()
    plot_df = combined_df[combined_df['Category'].isin(top_categories)]

    # Plotting
    plt.figure(figsize=(14, 10))
    ax = sns.barplot(data=plot_df, x='Count', y='Category', hue='Source', palette={'REAL': 'skyblue', 'GEN': 'salmon'})

    # Add annotations
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 1, p.get_y() + p.get_height() / 2,
                f'{int(width)}',
                va='center')

    plt.title(f'Vergleich der Kategorienverteilung\n(REAL: {total_real_images} Bilder, GEN: {total_gen_images} Bilder)', fontsize=16)
    plt.xlabel("Anzahl der Bilder", fontsize=12)
    plt.ylabel("Top 20 Kategorien (basierend auf REAL)", fontsize=12)
    plt.legend(title='Datenquelle')
    plt.tight_layout()
    plt.xlim(right=ax.get_xlim()[1] * 1.1) # Adjust x-axis limit for annotations

    # Save the plot
    plt.savefig(output_path)
    print(f"Successfully saved combined plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Erstellt ein kombiniertes Balkendiagramm der Kategorienverteilung.")
    parser.add_argument("--real-csv", type=str, default="reports/real_descriptions_categorized.csv", help="Pfad zur REAL CSV-Datei.")
    parser.add_argument("--gen-csv", type=str, default="reports/gen_descriptions_categorized.csv", help="Pfad zur GEN CSV-Datei.")
    parser.add_argument("--output-path", type=str, default="reports/combined_category_distribution.png", help="Pfad zur Speicherzieldatei für das Diagramm.")
    args = parser.parse_args()

    plot_combined_distribution(args.real_csv, args.gen_csv, args.output_path)

if __name__ == "__main__":
    main()

