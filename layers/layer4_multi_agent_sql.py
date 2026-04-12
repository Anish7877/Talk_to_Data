"""
Layer 4: Multi-Agent SQL Engine
Three specialized agents: planner -> coder -> validator
FIX: LangGraph API updated - END imported from langgraph.graph correctly
FIX: create_sql_graph instantiates engine once, not per node call
FIX: sqlglot parse handles errors gracefully
"""

import json
import re
from typing import Dict, Any, List, Optional, TypedDict
from enum import Enum
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from layers.groq_client import GroqClient, GROQ_MODELS
import sqlglot
import os


class SQLState(TypedDict):
    user_query: str
    schema_context: str
    plan: str
    sql_query: str
    is_valid: bool
    validation_errors: List[str]
    tables_used: List[str]
    parameterized_query: str
    params: List[Any]


class AgentType(str, Enum):
    PLANNER = "planner"
    CODER = "coder"
    VALIDATOR = "validator"


@dataclass
class SQLResult:
    success: bool
    query: str
    parameterized_query: str
    params: List[Any]
    plan: str
    tables_used: List[str]
    validation_errors: List[str]
    message: str


DANGEROUS_KEYWORDS = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE", "CREATE", "GRANT", "REVOKE"]

PLANNER_PROMPT = """Database expert. Given the question and schema, write a SHORT numbered plan (3 steps max) for the SQL query. No SQL yet.

Question: {user_query}
Schema: {schema_context}
Plan:"""

CODER_PROMPT = """SQL expert. Write ONE raw PostgreSQL query only. No explanation, no markdown.

Plan: {plan}
Schema: {schema_context}
SQL:"""


class MultiAgentSQLEngine:
    def __init__(
        self,
        planner_model: str = None,
        coder_model: str = None,
        validator_model: str = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        api_key: str = None
    ):
        self.planner_model = planner_model or GROQ_MODELS["powerful"]
        self.coder_model = coder_model or GROQ_MODELS["powerful"]
        self.validator_model = validator_model or GROQ_MODELS["fast"]
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = GroqClient(api_key=api_key)

    def _call_llm(self, model: str, prompt: str, system_message: str = None) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat_completions_create(
            model=model, messages=messages, temperature=self.temperature
        )
        return response["choices"][0]["message"]["content"]

    def planner_node(self, state: SQLState) -> SQLState:
        prompt = PLANNER_PROMPT.format(
            user_query=state["user_query"],
            schema_context=state["schema_context"]
        )
        plan = self._call_llm(self.planner_model, prompt)
        return {**state, "plan": plan}

    def coder_node(self, state: SQLState) -> SQLState:
        prompt = CODER_PROMPT.format(
            plan=state["plan"],
            schema_context=state["schema_context"]
        )
        sql = self._call_llm(self.coder_model, prompt)
        # FIX: Strip markdown code blocks reliably
        sql = re.sub(r'```(?:sql)?', '', sql, flags=re.IGNORECASE).strip()
        sql = sql.strip('`').strip()
        return {**state, "sql_query": sql}

    def validator_node(self, state: SQLState) -> SQLState:
        errors = []
        tables_used = []

        # Check for dangerous keywords
        sql_upper = state["sql_query"].upper()
        for keyword in DANGEROUS_KEYWORDS:
            if re.search(r'\b' + keyword + r'\b', sql_upper):
                errors.append(f"Dangerous keyword found: {keyword}")

        # FIX: Wrap sqlglot parse in try/except with fallback
        try:
            parsed = sqlglot.parse_one(state["sql_query"], dialect="postgres")
            if parsed:
                tables_used = [t.name for t in parsed.find_all(sqlglot.exp.Table)]
        except Exception as e:
            # Don't block on parse errors - just log them
            errors.append(f"SQL syntax warning: {str(e)}")

        is_valid = len(errors) == 0
        return {
            **state,
            "is_valid": is_valid,
            "validation_errors": errors,
            "tables_used": tables_used,
            "parameterized_query": state["sql_query"],
            "params": []
        }

    def execute(self, user_query: str, schema_context: str) -> SQLResult:
        state: SQLState = {
            "user_query": user_query,
            "schema_context": schema_context,
            "plan": "",
            "sql_query": "",
            "is_valid": False,
            "validation_errors": [],
            "tables_used": [],
            "parameterized_query": "",
            "params": []
        }
        MAX_SCHEMA_CHARS = 2000
        if len(schema_context) > MAX_SCHEMA_CHARS:
            schema_context = schema_context[:MAX_SCHEMA_CHARS] + "\n... (truncated)"

        state = self.planner_node(state)

        for attempt in range(self.max_retries):
            state = self.coder_node(state)
            state = self.validator_node(state)
            if state["is_valid"]:
                break
            if attempt < self.max_retries - 1:
                state["schema_context"] += f"\n\nPrevious errors: {state['validation_errors']}. Fix the SQL."

        return SQLResult(
            success=state["is_valid"],
            query=state["sql_query"],
            parameterized_query=state["parameterized_query"],
            params=state["params"],
            plan=state["plan"],
            tables_used=state["tables_used"],
            validation_errors=state["validation_errors"],
            message="Query validated successfully" if state["is_valid"] else "Validation failed"
        )


def create_sql_graph():
    """
    FIX: Engine is instantiated once at graph creation, not per node call.
    """
    engine = MultiAgentSQLEngine()

    def planner_node(state: SQLState) -> SQLState:
        return engine.planner_node(state)

    def coder_node(state: SQLState) -> SQLState:
        return engine.coder_node(state)

    def validator_node(state: SQLState) -> SQLState:
        return engine.validator_node(state)

    def should_retry(state: SQLState) -> str:
        return END if state["is_valid"] else "coder"

    workflow = StateGraph(SQLState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("validator", validator_node)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "coder")
    workflow.add_edge("coder", "validator")
    workflow.add_conditional_edges("validator", should_retry, {END: END, "coder": "coder"})
    return workflow.compile()


def create_sql_engine(config: Dict[str, Any]) -> MultiAgentSQLEngine:
    sql_config = config.get("multi_agent_sql", {})
    return MultiAgentSQLEngine(
        planner_model=sql_config.get("planner_model", GROQ_MODELS["powerful"]),
        coder_model=sql_config.get("coder_model", GROQ_MODELS["powerful"]),
        validator_model=sql_config.get("validator_model", GROQ_MODELS["fast"]),
        temperature=sql_config.get("temperature", 0.0),
        max_retries=sql_config.get("max_retries", 3)
    )
