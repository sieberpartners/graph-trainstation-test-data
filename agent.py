from langchain_core.messages import AnyMessage, ToolMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Dict, Any
from dotenv import load_dotenv
import operator

load_dotenv()

fachdatenmodell = """
    Nodes:
    - Origin(id, name)
    - Station(id, name, platform_count, opened_year)
    - City(id, name)
    - Country(id, name)
    Relationships:
    - (Station)-[:IS_IN]->(City)
    - (City)-[:IS_IN]->(Country)
    - (Station)-[:ORIGINALLY_FROM]->(Origin)
    - (City)-[:ORIGINALLY_FROM]->(Origin)
    - (Country)-[:ORIGINALLY_FROM]->(Origin)
"""

graph = Neo4jGraph()

llm = ChatOpenAI(model_name="gpt-4o-mini")
structured_llm = llm.with_structured_output(GraphDocument)

graph_doc_prompt = PromptTemplate.from_template("""
You are an AI agent tasked with transforming an input into a GraphDocument object based on a predefined schema.

The schema is as follows:
{fachdatenmodell}

The input is the following:
{input}
""")

structured_llm = graph_doc_prompt | structured_llm

@tool
def dict_to_graphdoc(original: Dict[str, Any]) -> GraphDocument:
    """Transform the input dictionary into a GraphDocument object based on the existing knowledge graph schema"""
    input_str = str(original)
    graph_document = structured_llm.invoke({"fachdatenmodell": fachdatenmodell, "input": input_str})
    return graph_document

@tool
def update_graph(query: str) -> str:
    """Update the Neo4j graph with the provided GraphDocument"""
    try:
        result = graph.query(query)
        return f"Graph extended successfully. Result: {result}"
    except Exception as e:
        return f"Error updating graph: {str(e)}"

@tool
def extend_graph(graph_document: GraphDocument) -> str:
    """Extend the graph by running a custom Cypher query"""
    try:
        graph.add_graph_documents([graph_document])
        return "Graph updated successfully"
    except Exception as e:
        return f"Error extending graph: {str(e)}"

tools = [dict_to_graphdoc, update_graph, extend_graph]

example_row = {"StationID": 1, "StationName": "Gare du Nord", "Location": "Paris, France", "Platforms": 36, "OpenedYear": 1864, "Origin": "trainstation_legacy1"}

# Define the state
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    current_row: Dict[str, Any]

class TrainstationAgent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("transform_input", self.transform_input)
        graph.add_node("llm", self.call_openai)
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
        return isinstance(result, AIMessage) and len(result.tool_calls) > 0

    def transform_input(self, state: AgentState):
        row = state['current_row']
        result = self.tools['dict_to_graphdoc'].invoke(input={"original": row})
        print(f"Transformed input: {result}")
        return {'messages': [AIMessage(tool_call_id=1, name='dict_to_graphdoc', content=str(result))]}

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
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
            elif t['name'] == 'dict_to_graphdoc':
                continue
            elif t['name'] == 'extend_graph':
                result = self.tools[t.name].invoke(input={"graph_document": state['messages'][1].content})
            else:
                result = self.tools[t.name].invoke(**t.args)
            results.append(ToolMessage(tool_call_id=t.id, name=t.name, content=str(result)))
        print("Back to the model!")
        return {'messages': results}

system_message = "You are an AI assistant processing train station data."

# Usage example
agent = TrainstationAgent(llm, tools, system=system_message)
result = agent.graph.invoke({"messages": [], "current_row": example_row})
print(result)