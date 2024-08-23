from langchain_core.messages import AnyMessage, ToolMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher_utils import Schema, CypherQueryCorrector
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Dict, Any
from dotenv import load_dotenv
from data import inventar_daten
import operator

load_dotenv()

fachdatenmodell = {
    "version": "1.0",
    "inhalt": """
        Nodes:
        - Origin(name)
        - Station(name, platform_count, opened_year)
        - City(name)
        - Country(name)
        - FDMVersion(number)
        Relationships:
        - (Station)-[:IS_IN]->(City)
        - (City)-[:IS_IN]->(Country)
        - (Station)-[:ORIGINALLY_FROM]->(Origin)
        - (City)-[:ORIGINALLY_FROM]->(Origin)
        - (Country)-[:ORIGINALLY_FROM]->(Origin)
        - (Station)-[:HAS_VERSION]->(FDMVersion)
        """
}

corrector_schema = [
    Schema("Station", "IS_IN", "City"),
    Schema("City", "IS_IN", "Country"),
    Schema("Station", "ORIGINALLY_FROM", "Origin"),
    Schema("City", "ORIGINALLY_FROM", "Origin"),
    Schema("Country", "ORIGINALLY_FROM", "Origin"),
    Schema("Station", "HAS_VERSION", "FDMVersion")
]

cypher_validation = CypherQueryCorrector(corrector_schema)

graph = Neo4jGraph()

llm = ChatOpenAI(model_name="gpt-4o-mini")

graph_doc_prompt = PromptTemplate.from_template("""
You are an AI agent tasked with transforming an input dictionary into a Cypher query based on a predefined graph schema and an existing knowledge graph.
The Cypher query is supposed to update the knowledge graph based on the information in the input dictionary.
Updating means adding, modifying or deleting nodes and relationships, depending on the input dictionary.
The knowledge graph and therefore the Cypher query have to respect the schema at all times.

The schema is as follows:
{fachdatenmodell}

The existing knowledge graph is as follows:
{existing_graph}

The input dictionary is the following:
{input}

Only return the cypher query. Do not describe anything.
""")

cypher_llm = graph_doc_prompt | llm

@tool
def dict_to_cypher(original: Dict[str, Any]) -> str:
    """Transform the input dictionary into a Cypher query based on the existing knowledge graph schema."""
    input_str = str(original)
    cypher_query = cypher_llm.invoke({"fachdatenmodell": fachdatenmodell["inhalt"], "existing_graph": graph.get_schema, "input": input_str})
    return cypher_query

@tool
def update_graph(query: str) -> str:
    """Update existing nodes and relationships in the neo4j graph with the provided Cypher query"""
    try:
        result = graph.query(cypher_validation(query))
        return f"Graph updated successfully. Result: {result}"
    except Exception as e:
        return f"Error updating graph: {str(e)}"

tools = [dict_to_cypher, update_graph]

# Define the state
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    current_row: Dict[str, Any]

class TrainstationAgent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("transform_input", self.transform_input)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_edge("transform_input", "llm")
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("transform_input")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def transform_input(self, state: AgentState):
        result = self.tools['dict_to_cypher'].invoke(input={"original": state['current_row']})
        print(f"Transformed input: {result}")
        return {'messages': [result]}

    def call_llm(self, state: AgentState):
        messages = state['messages']
        if self.system and type(messages[0]) != SystemMessage:
            messages.insert(0, SystemMessage(content=self.system))
        message = self.model.invoke(messages)
        return {'messages': [message]}
    
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if t['name'] not in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            elif t['name'] == 'dict_to_cypher':
                result = self.tools['dict_to_cypher'].invoke(input={"original": state['current_row']})
            elif t['name'] == 'update_graph':
                result = self.tools['update_graph'].invoke(input={"query": t['args']['query']})
                print(f"Query: {t['args']['query']}")
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

system_message = "You are an AI assistant processing train station data."

agent = TrainstationAgent(llm, tools, system=system_message)

# Usage example
""" example_row_1 = {"StationID": 1, "StationName": "Gare du Nord", "Location": "Paris, France", "Platforms": 36, "OpenedYear": 1864, "Origin": "trainstation_legacy1"}
example_row_2 = {"ID": 1, "Nom": "Gare du Nord", "Ville": "Paris", "Pays": "France", "Plateformes": 36, "Ouverture": 1864, "Origine": "trainstation_legacy2", "Coordonnees": "48.8809, 2.3550"}
result = agent.graph.invoke({"messages": [], "current_row": example_row_1}) """

counter = 0
for inventar_tabelle in inventar_daten:
    name = "dfa" if 0 < counter < 11 else "itop" if 11 < counter < 22 else "legacy_system_x"
    for _, row in inventar_tabelle.iterrows():
        if counter == 3 or counter == 13 or counter == 28:
            row["Origin"] = name
            row["FDM-Version"] = fachdatenmodell["version"]
            result = agent.graph.invoke({"messages": [], "current_row": row})
        counter += 1