import pandas as pd
import os

# נתיב התיקייה עם קבצי האקסל
folder_path = f"C:\\Users\\ASUS\\Desktop\\SoftwareEngineering\\שנה ד\\פרוייקט גמר\\POC\\FeedLect"

# רשימת DataFrames לאיחוד
dataframes = []

for file in os.listdir(folder_path):
    if file.endswith(".xlsx"):  # בדיקה אם הקובץ הוא אקסל
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path)  # קריאת הנתונים מהקובץ
        df['Source File'] = file  # הוספת עמודה עם שם הקובץ
        dataframes.append(df)

# איחוד כל ה-DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# שמירת הקובץ המאוחד
output_file = "merged.xlsx"
merged_df.to_excel(output_file, index=False)

print(f"כל הקבצים אוחדו לגיליון אחד בתוך {output_file}")