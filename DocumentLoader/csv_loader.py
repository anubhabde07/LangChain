from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='annual-enterprise-survey-2024-financial-year-provisional.csv')

data = loader.load()
print(len(data))
print(data[0])