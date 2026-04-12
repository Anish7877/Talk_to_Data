"""
Layer 2: Intent Router
Classifies queries into: SQL, RAG, or Both routes
FIX: Removed duplicate IntentRouter class definition
FIX: Removed ROUTER_PROMPT from the first (deleted) class that was never used
"""

import json
from typing import Literal, Dict, Any, TypedDict
from enum import Enum
from langgraph.graph import StateGraph, END
from layers.groq_client import GroqClient, GROQ_MODELS
import os


class RouteType(str, Enum):
    SQL = "sql"
    RAG = "rag"
    BOTH = "both"


class RouterState(TypedDict):
    query: str
    route: str
    confidence: float
    reasoning: str


class IntentRouter:
    """
    Routes user queries to appropriate pipelines using a fast LLM via Groq.
    FIX: Removed duplicate class definition that existed in original file.
    """

    ROUTER_PROMPT = """You are a query classifier and intent routing agent for an AI system.

    Given a user query, classify it into one of three categories:
    - "sql": The query asks for structured data, metrics, counts, or numerical data from a database
    - "rag": The query asks for explanations, definitions, concepts, documents, policies, or unstructured text
    - "both": The query requires both database data and document context

    AND to identify any required database schemas or tables the user implies.
    If the user explicitly uses a mention like `@tablename`, capture it. If they don't, figure out the logical schema/tables they are asking about based on context (e.g. if they say 'sales', they might mean 'orders').

    Return a JSON object with:
    - "route": The category ("sql", "rag", or "both")
    - "schemas": A list of strings of database tables/schemas needed (e.g. ["customers", "orders"])
    - "confidence": A number between 0 and 1
    - "reasoning": Brief explanation of the classification

    Examples:
    - "How many orders did we have last month?" -> {{"route": "sql", "schemas": ["orders"], ...}}
    - "What is our return policy?" -> {{"route": "rag", "schemas": [], ...}}
    - "Show me sales by region" -> {{"route": "sql", "schemas": ["orders", "sales", "region"], ...}}
    - "Get details about @users" -> {{"route": "sql", "schemas": ["users"], ...}}

    Query: {query}

    Respond with JSON only."""

    def __init__(self, model: str = None, temperature: float = 0.0, api_key: str = None):
        self.model = model or GROQ_MODELS["fast"]
        self.temperature = temperature
        self.client = GroqClient(api_key=api_key)

    def route(self, query: str) -> Dict[str, Any]:
        response = self.client.chat_completions_create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a query classifier. Return only valid JSON."},
                {"role": "user", "content": self.ROUTER_PROMPT.format(query=query)}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )

        result = json.loads(response["choices"][0]["message"]["content"])
        return {
            "route": result.get("route", "sql"),
            "schemas": result.get("schemas", []),
            "confidence": result.get("confidence", 0.5),
            "reasoning": result.get("reasoning", "")
        }

    def route_sync(self, query: str) -> Literal["sql", "rag", "both"]:
        return self.route(query)["route"]


def create_router_graph():
    def route_node(state: RouterState) -> RouterState:
        router = IntentRouter()
        result = router.route(state["query"])
        return {**state, "route": result["route"], "confidence": result["confidence"], "reasoning": result["reasoning"]}

    def sql_pipeline_node(state: RouterState) -> RouterState:
        return {**state, "sql_result": "sql_result_placeholder"}

    def rag_pipeline_node(state: RouterState) -> RouterState:
        return {**state, "rag_result": "rag_result_placeholder"}

    def both_pipeline_node(state: RouterState) -> RouterState:
        return {**state, "sql_result": "sql_result_placeholder", "rag_result": "rag_result_placeholder"}

    def should_continue(state: RouterState) -> str:
        route = state.get("route", "sql")
        if route == "sql":
            return "sql_pipeline"
        elif route == "rag":
            return "rag_pipeline"
        return "both_pipeline"

    workflow = StateGraph(RouterState)
    workflow.add_node("route", route_node)
    workflow.add_node("sql_pipeline", sql_pipeline_node)
    workflow.add_node("rag_pipeline", rag_pipeline_node)
    workflow.add_node("both_pipeline", both_pipeline_node)
    workflow.set_entry_point("route")
    workflow.add_conditional_edges("route", should_continue, {
        "sql_pipeline": "sql_pipeline",
        "rag_pipeline": "rag_pipeline",
        "both_pipeline": "both_pipeline"
    })
    workflow.add_edge("sql_pipeline", END)
    workflow.add_edge("rag_pipeline", END)
    workflow.add_edge("both_pipeline", END)
    return workflow.compile()


def create_intent_router(config: Dict[str, Any]) -> IntentRouter:
    router_config = config.get("intent_router", {})
    return IntentRouter(
        model=router_config.get("model", GROQ_MODELS["fast"]),
        temperature=router_config.get("temperature", 0.0)
    )


if __name__ == "__main__":
    router = IntentRouter()
    test_queries = [
        "How many orders did we have last month?",
        "What is our return policy?",
        "Show me sales by region and explain our compensation policy",
    ]
    print("Intent Routing Examples:")
    print("-" * 50)
    for query in test_queries:
        result = router.route(query)
        print(f"Query: {query}")
        print(f"Route: {result['route']} (confidence: {result['confidence']:.2f})")
        print(f"Reasoning: {result['reasoning']}")
        print("-" * 50)
