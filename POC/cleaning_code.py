import pandas as pd

# Correct the file path
file_path = r'C:\4rd_year\final_project\all_data\pythonProject1\שאלונים תשפד.xlsx'

# Load the Excel file
data = pd.read_excel(file_path, sheet_name='Default')

# All columns that you want to check for missing data
potential_columns_to_clean = [
    "ממוצע שאלה  1",
    "ממוצע שאלה  2",
    "ממוצע שאלה  3",
    "ממוצע שאלה  4",
    "תשובה מילולית כ/ל"
]

# Filter only the columns that exist in the DataFrame
columns_to_clean = [col for col in potential_columns_to_clean if col in data.columns]

# Drop rows with missing values in the existing columns
cleaned_data = data.dropna(subset=columns_to_clean)

# Save the cleaned data to a new Excel file
output_path = r'C:\4rd_year\final_project\all_data\pythonProject1\cleaned_data\שאלונים תשפד מנוקה.xlsx'
cleaned_data.to_excel(output_path, index=False)

print(f"Cleaned data saved to: {output_path}")
