import operator
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher_utils import Schema, CypherQueryCorrector
from langchain_anthropic import ChatAnthropic 
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition, InjectedState
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    existing_graph: str

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
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

base_prompt = """
You transform an input dictionary into a Cypher query to {action} an existing knowledge graph based on a predefined graph schema.
Your task is to generate a Cypher query that {task_description}.

The schema is as follows: {fachdatenmodell}
The existing knowledge graph is as follows: {existing_graph}
The input data {input_description} is the following: {input_data}
Follow these rules to generate the Cypher query:

{rules}

Only return the Cypher query. Do not describe anything.
"""

extend_prompt = PromptTemplate.from_template(base_prompt).partial(
    action="extend",
    task_description="adds new nodes and relationships to the graph",
    input_description="to be added",
    rules="""
Generate a Cypher query to add new nodes and relationships based on the input data.
Use MERGE for nodes to avoid duplicates.
Use CREATE for relationships if they don't exist.
Ensure that all operations respect the predefined schema at all times.
"""
)

update_prompt = PromptTemplate.from_template(base_prompt).partial(
    action="update",
    task_description="modifies existing nodes and relationships in the graph",
    input_description="for updating",
    rules="""
Generate a Cypher query to modify existing nodes or relationships based on the input data.
Use MATCH to find the existing nodes/relationships.
Use SET to update properties.
If necessary, use CREATE to add new relationships between existing nodes.
Ensure that all operations respect the predefined schema at all times.
"""
)

delete_prompt = PromptTemplate.from_template(base_prompt).partial(
    action="delete parts of",
    task_description="removes specified nodes and relationships from the graph",
    input_description="specifying what to delete",
    rules="""
Generate a Cypher query to delete nodes or relationships based on the input data.
Use MATCH to find the nodes/relationships to be deleted.
Use DETACH DELETE for nodes (to remove connected relationships) or DELETE for relationships.
Ensure that all operations respect the predefined schema at all times.
Be cautious with deletions to maintain graph integrity.
"""
)

@tool
def extend_graph(input_data: str, state: Annotated[dict, InjectedState]) -> str:
    """Extend the Neo4j graph with new nodes and relationships based on the input data."""
    chain = extend_prompt | llm | StrOutputParser() | cypher_validation
    cypher_query = chain.invoke({
        "fachdatenmodell": fachdatenmodell["inhalt"], 
        "existing_graph": state["existing_graph"], 
        "input_data": input_data
    })
    result = graph.query(cypher_query)
    return f"Graph extended successfully. Result: {result}"

@tool
def update_graph(input_data: str, state: Annotated[dict, InjectedState]) -> str:
    """Update existing nodes and relationships in the Neo4j graph based on the input data."""
    chain = update_prompt | llm | cypher_validation
    cypher_query = chain.invoke({
        "fachdatenmodell": fachdatenmodell["inhalt"], 
        "existing_graph": state["existing_graph"], 
        "input_data": input_data
    })
    result = graph.query(cypher_query)
    return f"Graph updated successfully. Result: {result}"

@tool
def delete_graph(input_data: str, state: Annotated[dict, InjectedState]) -> str:
    """Delete specified nodes and relationships from the Neo4j graph based on the input data."""
    chain = delete_prompt | llm | cypher_validation
    cypher_query = chain.invoke({
        "fachdatenmodell": fachdatenmodell["inhalt"], 
        "existing_graph": state["existing_graph"], 
        "input_data": input_data
    })
    result = graph.query(cypher_query)
    return f"Graph deleted successfully. Result: {result}"

tools = [extend_graph, update_graph, delete_graph]
tool_node = ToolNode(tools)
bound_model = llm.bind_tools(tools)

system_prompt = """
You are an intelligent system responsible for analyzing input data and deciding how to modify a knowledge graph.
Your task is to determine whether the given input should result in extending, updating, or deleting parts of an existing graph.
You will be provided with the following information:

A predefined data schema: {fachdatenmodell}
The existing knowledge graph: {existing_graph}

Based on this information, you must consider the input data and decide on the appropriate action to take.
Carefully compare the input data with the existing graph.
Consider the graph schema to ensure your decision will maintain the integrity of the graph structure.
If the input contains a mix of new and existing information, update rather than extend.
Only delete something when there's a clear indication that information should be removed.
If you're unsure, lean towards update as it's generally safer than delete.
"""

def create_system_message(fachdatenmodell, existing_graph):
    return SystemMessage(content=system_prompt.format(
        fachdatenmodell=fachdatenmodell,
        existing_graph=existing_graph
    ))

def reflect(state: State):
    messages = state["messages"]
    
    if not messages or not isinstance(messages[0], SystemMessage):
        system_message = create_system_message(fachdatenmodell, state["existing_graph"])
        messages = [system_message] + messages
        
    response = bound_model.invoke(messages)
    return {"messages": [response], "existing_graph": graph.get_schema if graph.get_schema else 'Empty graph.'}

workflow = StateGraph(State)
workflow.add_node("reflect", reflect)
workflow.add_node("act", tool_node)
workflow.add_edge(START, "reflect")
workflow.add_conditional_edges("reflect", tools_condition, {"tools": "act", END: END})
workflow.add_edge("act", "reflect")
langgraph = workflow.compile()

# Usage example
example_row_1 = {"StationID": 1, "StationName": "Gare du Nord", "Location": "Paris, France", "Platforms": 36, "OpenedYear": 1864, "Origin": "trainstation_legacy1"}
human_message = HumanMessage(content=str(example_row_1))
print(langgraph.invoke({"messages": [human_message]}))