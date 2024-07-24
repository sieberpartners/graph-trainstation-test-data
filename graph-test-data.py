import pandas as pd
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

# Creating the synthetic test legacy data
trainstation_legacy1 = {
    "StationID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "StationName": ["Gare du Nord", "St Pancras", "Hauptbahnhof", "Centrale", "Atocha", "Sants", "Zurich HB", "Amsterdam Centraal", "Wien Hauptbahnhof", "Gare de Lyon", "München Hbf", "Antwerp Central"],
    "Location": ["Paris, France", "London, UK", "Berlin, Germany", "Milan, Italy", "Madrid, Spain", "Barcelona, Spain", "Zurich, Switzerland", "Amsterdam, Netherlands", "Vienna, Austria", "Paris, France", "Munich, Germany", "Antwerp, Belgium"],
    "Platforms": [36, 15, 16, 24, 21, 14, 26, 15, 12, 13, 32, 14],
    "OpenedYear": [1864, 1868, 1871, 1931, 1851, 1975, 1847, 1889, 2012, 1900, 1849, 1905]
}

trainstation_legacy2 = {
    "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Name": ["Gare du Nord", "St Pancras", "Hauptbahnhof", "Centrale", "Atocha", "Sants", "Zurich HB", "Amsterdam Centraal", "Wien Hauptbahnhof", "Gare de Lyon", "München Hbf", "Antwerp Central"],
    "City": ["Paris", "London", "Berlin", "Milan", "Madrid", "Barcelona", "Zurich", "Amsterdam", "Vienna", "Paris", "Munich", "Antwerp"],
    "Country": ["France", "UK", "Germany", "Italy", "Spain", "Spain", "Switzerland", "Netherlands", "Austria", "France", "Germany", "Belgium"],
    "NumberOfPlatforms": [36, 15, 16, 24, 21, 14, 26, 15, 12, 13, 32, 14],
    "YearOpened": [1864, 1868, 1871, 1931, 1851, 1975, 1847, 1889, 2012, 1900, 1849, 1905]
}

trainstation_legacy3 = {
    "StationCode": ["FRPAR", "UKLON", "DEBER", "ITMIL", "ESMAD", "ESBCN", "CHZRH", "NLAMS", "ATVIE", "FRPAR2", "DEMUC", "BEBRU"],
    "FullName": ["GARE DU NORD", "ST PANCRAS", "HAUPTBAHNHOF", "CENTRALE", "ATOCHA", "SANTS", "ZURICH HB", "AMSTERDAM CENTRAAL", "WIEN HAUPTBAHNHOF", "GARE DE LYON", "MÜNCHEN HBF", "ANTWERP CENTRAL"],
    "Address": ["PARIS, FRANCE", "LONDON, UK", "BERLIN, GERMANY", "MILAN, ITALY", "MADRID, SPAIN", "BARCELONA, SPAIN", "ZURICH, SWITZERLAND", "AMSTERDAM, NETHERLANDS", "VIENNA, AUSTRIA", "PARIS, FRANCE", "MUNICH, GERMANY", "ANTWERP, BELGIUM"],
    "PlatformsCount": ['36', '15', '16', '24', '21', '14', '26', '15', '12', '13', '32', '14'],
    "Established": ['1864', '1868', '1871', '1931', '1851', '1975', '1847', '1889', '2012', '1900', '1849', '1905']
}

# Creating the synthetic test fachdatenmodell
trainstation_fachdatenmodell = ['ID', 'StationName', 'City', 'Country', 'Platforms', 'OpenedYear']

documents = [Document(page_content=str(legacy_data)) for legacy_data in [trainstation_legacy1, trainstation_legacy2, trainstation_legacy3]]

instruction = """
You are an expert in train data. You have been given three legacy datasets that contain information about train stations.
You are to create a knowledge graph based on allowed nodes and the legacy data. Keep as much information as possible from the legacy data.
Create a node for each train station, country and city. Give the train station properties such as name, platforms andd year opened to store more info.
"""
prompt = ChatPromptTemplate.from_template(instruction)

# Instantiate the transformer with the LLM model
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=trainstation_fachdatenmodell,
    allowed_relationships=["is_in", "has", "opened_in"],
    strict_mode=True,
    )

graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")

from langchain_community.graphs import Neo4jGraph

load_dotenv()

graph = Neo4jGraph()

graph.add_graph_documents(graph_documents)