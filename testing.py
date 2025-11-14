
# from docling.document_converter import DocumentConverter
# converter = DocumentConverter()
# all_sheets_df = pd.DataFrame()
# for sheet_name in xls.sheet_names:
#     print(f"  - Reading sheet: {sheet_name}")
#     df = pd.read_excel(xls, sheet_name=sheet_name, header=2)
#     print(df)
#     all_sheets_df = pd.concat([all_sheets_df, df], ignore_index=True)

# csv_file_path = "./csv_temp.csv"
# all_sheets_df.to_csv(csv_file_path, index=False)
# file_path = csv_file_path
# print(f"Successfully converted Excel to temporary CSV: {file_path}")
# result = converter.convert(file_path)
# output_content = result.document.export_to_markdown()

# print(output_content)

# import pandas as pd
# from openpyxl import load_workbook
# from json import dumps
# 
# sheets = []
# for sheet_name in xls.sheet_names:
# # Load Excel workbook

#     # Choose a specific sheet
#     workbook = load_workbook(filename="./compare-institutions-2025-11-14-5-42.xlsx")
#     sheet = workbook[sheet_name]
#     # Find the number of rows and columns in the sheet
#     rows = sheet.max_row
#     columns = sheet.max_column

#     # List to store all rows as dictionaries
#     lst = []

#     # Iterate over rows and columns to extract data
#     for i in range(1, rows):
#         row = {}
#         for j in range(1, columns):
#             column_name = sheet.cell(row=1, column=j)
#             row_data = sheet.cell(row=i+1, column=j)

#             row.update(
#                 {
#                     column_name.value: row_data.value
#                 }
#             )
#         lst.append(row)
#     sheet.append(lst)
# # Convert extracted data into JSON format
# json_data = dumps(sheet)

# # Print the JSON data
# print(json_data)


# import pandas as pd
# import json
# json_list = []
# # Convert Excel To Json With Python
# xls = pd.ExcelFile("./compare-institutions-2025-11-14-5-42.xlsx")
# for sheet_name in xls.sheet_names:
#     df = pd.read_excel(xls, sheet_name=sheet_name, header=3)
#     jdf = df.dropna(how="all")
#     # JSON come lista di record: [{col1:..., col2:...}, ...]
#     records = df.to_dict(orient="records")
#     json_list.append({sheet_name: records})

# json_output = json.dumps(json_list, indent=4)
# print(json_output)

from excel2json import convert_from_file

# Convert Excel file to JSON
json_file = convert_from_file("compare-institutions-2025-11-14-5-42.xlsx")
print(json_file)