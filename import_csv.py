import csv
from collections import Counter

def count_unique_values_in_column(file_path, column_name, output_file):
    print(f"Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        print("Available columns:", reader.fieldnames)

        if column_name not in reader.fieldnames:
            print(f"Column '{column_name}' not found.")
            return

        values = [row[column_name] for row in reader if row[column_name]]
        total = len(values)
        counter = Counter(values)

        # Sort by count descending
        sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        print(f"\nWriting sorted results with percentages to: {output_file}")
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([column_name, "count", "percentage"])
            for value, count in sorted_counts:
                percentage = (count / total) * 100
                writer.writerow([value, count, f"{percentage:.2f}%"])

        print("Done!")

if __name__ == "__main__":
    print("Script started")
    file_path = "EOD_cdw_tmds_diagnoses_summary_11292025.tsv"
    column_name = "diagnosis"
    output_file = "diagnosis_counts.csv"
    count_unique_values_in_column(file_path, column_name, output_file)
