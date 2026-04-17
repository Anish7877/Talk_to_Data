""" Layer 6: Storyteller & Lineage Engine Natural language answers with full audit trail """

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from layers.groq_client import GroqClient, GROQ_MODELS
import os
from pathlib import Path


@dataclass
class LineageTrace:
    """Audit trail for every query execution."""
    query: str
    route: str
    sql_run: Optional[str]
    tables_used: List[str]
    schemas_retrieved: List[str]
    documents_retrieved: List[str]
    cache_hit: bool
    cache_similarity: Optional[float]
    execution_time_ms: float
    timestamp: str
    user_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class Storyteller:
    """
    Generates natural language answers from query results.
    Includes full lineage tracing for audit purposes.
    """

    SYSTEM_PROMPT = """You are Nexus Intelligence, a strict, factual enterprise data assistant.
    Your task is to answer the user's question using ONLY the provided SQL Results and Document Context.

    ### STRICT RULES:
    1. NO EXTERNAL KNOWLEDGE: You must absolutely NEVER use your pre-trained knowledge to answer the question.
    2. GROUNDING: If the answer cannot be explicitly found in the SQL Results or Document Context provided below, you MUST reply with: "I do not have enough information in the current data to answer that."
    3. NO GUESSING: Do not extrapolate, guess, or assume missing data.
    4. FORMATTING: Be concise. If SQL data is provided, state the numbers clearly. Do not mention SQL or database tables in your final answer.
    """

    USER_PROMPT = """
    ### User Question:
    {user_question}

    ### SQL Results (Structured Data):
    {sql_results}

    ### Document Context (Unstructured Data):
    {doc_context}

    Answer the question strictly following the rules above.
    """

    def __init__(
        self,
        model: str = None,
        temperature: float = 0.3,
        max_sentences: int = 3,
        api_key: str = None,
        lineage_log_path: str = "./data/lineage_logs.jsonl"
    ):
        """
        Initialize the storyteller.

        Args:
            model: LLM model for generating answers (defaults to powerful Groq model)
            temperature: Sampling temperature
            max_sentences: Maximum sentences in answer
            api_key: Groq API key
            lineage_log_path: Path for lineage log file
        """
        self.model = model or GROQ_MODELS["powerful"]
        self.temperature = temperature
        self.max_sentences = max_sentences
        self.client = GroqClient(api_key=api_key)
        self.lineage_log_path = Path(lineage_log_path)

        # Ensure log directory exists
        self.lineage_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def _format_sql_results(self, rows: List[Dict[str, Any]]) -> str:
        """Format SQL results for the prompt."""
        if not rows:
            return "No results found."

        # Limit to first 10 rows for prompt
        display_rows = rows[:10]
        formatted = []

        for row in display_rows:
            row_str = ", ".join(f"{k}: {v}" for k, v in row.items())
            formatted.append(row_str)

        if len(rows) > 10:
            formatted.append(f"... and {len(rows) - 10} more rows")

        return "\n".join(formatted)

    def _format_doc_context(self, docs: List[Dict[str, Any]]) -> str:
        """Format document context for the prompt."""
        if not docs:
            return "No document context available."

        formatted = []
        for i, doc in enumerate(docs[:5], 1):
            content = doc.get("content", "")[:500]  # Limit content
            formatted.append(f"[{i}] {content}")

        return "\n\n".join(formatted)

    def _generate_answer(
        self,
        prompt: str,
        system_message: str = None
    ) -> str:
        """Generate answer using Groq API."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat_completions_create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=500
        )

        return response["choices"][0]["message"]["content"]

    def tell(
        self,
        user_question: str,
        sql_results: Optional[List[Dict[str, Any]]] = None,
        doc_context: Optional[List[Dict[str, Any]]] = None,
        route: str = "sql"
    ) -> str:
        """
        Generate a natural language answer with strict hallucination guardrails.
        """
        # Format the incoming data, providing safe fallback text if empty
        formatted_sql = self._format_sql_results(sql_results) if sql_results else "No SQL data retrieved or query failed."
        formatted_doc = self._format_doc_context(doc_context) if doc_context else "No document context retrieved."

        # Assemble the user prompt
        prompt = self.USER_PROMPT.format(
            user_question=user_question,
            sql_results=formatted_sql,
            doc_context=formatted_doc
        )

        # Generate the answer by explicitly passing the SYSTEM_PROMPT to the LLM
        return self._generate_answer(prompt=prompt, system_message=self.SYSTEM_PROMPT)

    def log_lineage(self, trace: LineageTrace) -> bool:
        """
        Log a lineage trace to file.

        Args:
            trace: LineageTrace object

        Returns:
            True if logged successfully
        """
        try:
            with open(self.lineage_log_path, "a") as f:
                f.write(trace.to_json() + "\n")
            return True
        except Exception as e:
            self.logger.error(f"Failed to log lineage: {str(e)}")
            return False

    def get_lineage_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent lineage logs."""
        logs = []
        if os.path.exists(self.lineage_log_path):
            import json  # Make sure json is imported
            try:
                with open(self.lineage_log_path, "r") as f:
                    lines = f.readlines()
                    # Parse the JSON string from each line back into a dictionary
                    for line in reversed(lines[-limit:]):
                        try:
                            logs.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                pass
        return logs

    def create_lineage(
        self,
        query: str,
        route: str,
        sql_query: Optional[str] = None,
        tables_used: Optional[List[str]] = None,
        schemas_retrieved: Optional[List[str]] = None,
        documents_retrieved: Optional[List[str]] = None,
        cache_hit: bool = False,
        cache_similarity: Optional[float] = None,
        execution_time_ms: float = 0
    ) -> LineageTrace:
        """Create a lineage trace for the current query."""
        return LineageTrace(
            query=query,
            route=route,
            sql_run=sql_query,
            tables_used=tables_used or [],
            schemas_retrieved=schemas_retrieved or [],
            documents_retrieved=documents_retrieved or [],
            cache_hit=cache_hit,
            cache_similarity=cache_similarity,
            execution_time_ms=execution_time_ms,
            timestamp=datetime.utcnow().isoformat()
        )


@dataclass
class QueryResponse:
    """Complete response from the query system."""
    answer: str
    lineage: LineageTrace
    raw_results: Optional[List[Dict[str, Any]]] = None
    raw_docs: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "lineage": self.lineage.to_dict(),
            "raw_results": self.raw_results,
            "raw_docs": self.raw_docs
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# Factory function
def create_storyteller(config: Dict[str, Any]) -> Storyteller:
    """Create a Storyteller from configuration."""
    storyteller_config = config.get("storyteller", {})
    return Storyteller(
        model=storyteller_config.get("model", GROQ_MODELS["powerful"]),
        temperature=storyteller_config.get("temperature", 0.3),
        max_sentences=storyteller_config.get("max_sentences", 3),
        lineage_log_path=config.get("logging", {}).get("lineage_log_path", "./data/lineage_logs.jsonl")
    )


if __name__ == "__main__":
    # Example usage
    storyteller = Storyteller()

    # Example SQL results
    sql_results = [
        {"region": "North America", "total_revenue": 1500000, "order_count": 12500},
        {"region": "Europe", "total_revenue": 1200000, "order_count": 9800},
        {"region": "Asia Pacific", "total_revenue": 900000, "order_count": 7500}
    ]

    # Generate answer
    answer = storyteller.tell(
        user_question="Show me revenue by region",
        sql_results=sql_results,
        route="sql"
    )

    print("Storyteller Example:")
    print("-" * 50)
    print(f"Question: Show me revenue by region")
    print(f"Answer: {answer}")
    print()

    # Create and log lineage
    lineage = storyteller.create_lineage(
        query="Show me revenue by region",
        route="sql",
        sql_query="SELECT region, SUM(total_amount) FROM orders GROUP BY region",
        tables_used=["orders", "customers"],
        schemas_retrieved=["orders"],
        execution_time_ms=250
    )

    print(f"Lineage trace:")
    print(lineage.to_json())
