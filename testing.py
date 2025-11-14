from docling.document_converter import DocumentConverter

source = "./US News: Profile.pdf"  # file path or URL
converter = DocumentConverter()
doc = converter.convert(source).document
tables = doc.tables

for table in tables:
    print(table.export_to_markdown())