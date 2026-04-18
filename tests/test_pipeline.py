"""
Tests for AI Query System
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json


class TestSemanticCache:
    """Tests for Layer 1: Semantic Cache."""

    def test_cache_initialization(self):
        """Test cache can be initialized."""
        from layers import SemanticCache

        with patch('layers.layer1_semantic_cache.Redis'):
            cache = SemanticCache(
                redis_host="localhost",
                redis_port=6379
            )
            assert cache.similarity_threshold == 0.92
            assert cache.ttl_seconds == 3600

    def test_cache_get_returns_none_when_empty(self):
        """Test cache returns None for empty cache."""
        from layers import SemanticCache

        with patch('layers.layer1_semantic_cache.Redis') as mock_redis:
            mock_instance = MagicMock()
            mock_instance.scan_iter.return_value = []
            mock_redis.return_value = mock_instance

            cache = SemanticCache()
            result = cache.get("test query")
            assert result is None

    def test_cache_stats(self):
        """Test cache statistics."""
        from layers import SemanticCache

        with patch('layers.layer1_semantic_cache.Redis') as mock_redis:
            mock_instance = MagicMock()
            mock_instance.scan_iter.return_value = ["cache:abc", "cache:def"]
            mock_redis.return_value = mock_instance

            cache = SemanticCache()
            stats = cache.get_stats()

            assert "total_entries" in stats
            assert stats["total_entries"] == 2


class TestIntentRouter:
    """Tests for Layer 2: Intent Router."""

    def test_router_initialization(self):
        """Test router can be initialized."""
        from layers import IntentRouter

        router = IntentRouter(model="gpt-4o-mini")
        assert router.model == "gpt-4o-mini"
        assert router.temperature == 0.0

    def test_route_return_types(self):
        """Test router returns valid route types."""
        from layers import IntentRouter, RouteType

        valid_routes = ["sql", "rag", "both"]
        for route in valid_routes:
            assert route in [r.value for r in RouteType]


class TestTAGRetrieval:
    """Tests for Layer 3: TAG Retrieval."""

    def test_table_description(self):
        """Test table description creation."""
        from layers import TableDescription

        table = TableDescription(
            table_name="test_table",
            description="Test table description",
            columns=[
                {"name": "id", "type": "INTEGER"},
                {"name": "name", "type": "VARCHAR"}
            ]
        )

        assert table.table_name == "test_table"
        assert len(table.columns) == 2

    def test_table_to_document(self):
        """Test table description to document conversion."""
        from layers import TableDescription

        table = TableDescription(
            table_name="customers",
            description="Customer information",
            columns=[
                {"name": "customer_id", "type": "INTEGER"}
            ]
        )

        doc = table.to_document()
        assert "Table: customers" in doc
        assert "customer_id" in doc


class TestMultiAgentSQL:
    """Tests for Layer 4: Multi-Agent SQL Engine."""

    def test_sql_result_dataclass(self):
        """Test SQLResult dataclass."""
        from layers import SQLResult

        result = SQLResult(
            success=True,
            query="SELECT * FROM customers",
            parameterized_query="SELECT * FROM customers",
            params=[],
            plan="Step 1: Select all customers",
            tables_used=["customers"],
            validation_errors=[],
            message="Success"
        )

        assert result.success is True
        assert result.tables_used == ["customers"]

    def test_validator_dangerous_keywords(self):
        """Test validator detects dangerous keywords."""
        from layers.layer4_multi_agent_sql import MultiAgentSQLEngine, DANGEROUS_KEYWORDS

        # Mock the engine (we don't need real API keys just to test validation logic)
        engine = MultiAgentSQLEngine(api_key="dummy_key")

        # These should be detected as dangerous
        dangerous_queries = [
            "DROP TABLE customers",
            "DELETE FROM orders",
            "INSERT INTO users VALUES (1, 'test')",
            "UPDATE products SET price = 0"
        ]

        for query in dangerous_queries:
            # Create a mock LangGraph state
            state = {
                "user_query": "test",
                "schema_context": "",
                "plan": "",
                "sql_query": query,
                "is_valid": True,
                "validation_errors": [],
                "tables_used": [],
                "parameterized_query": "",
                "params": []
            }

            # Run the validation node
            result_state = engine.validator_node(state)

            # Assert that the query was flagged as invalid
            assert result_state["is_valid"] is False
            assert len(result_state["validation_errors"]) > 0

            # Assert that the error message mentions "Dangerous keyword"
            assert any("Dangerous keyword found" in err for err in result_state["validation_errors"])

class TestSecureExecution:
    """Tests for Layer 5: Secure Execution."""

    def test_execution_result(self):
        """Test ExecutionResult dataclass."""
        from layers import ExecutionResult

        result = ExecutionResult(
            success=True,
            rows=[{"id": 1, "name": "Test"}],
            row_count=1,
            columns=["id", "name"],
            execution_time_ms=50.0
        )

        assert result.success is True
        assert result.row_count == 1
        assert len(result.columns) == 2

    def test_role_manager_sql_generation(self):
        """Test role manager generates correct SQL."""
        from layers import DatabaseRoleManager

        sql = DatabaseRoleManager.create_readonly_role_sql("test_role")

        assert len(sql) > 0
        assert "CREATE ROLE test_role" in sql[0]
        assert "GRANT SELECT" in sql[2]


class TestStoryteller:
    """Tests for Layer 6: Storyteller."""

    def test_lineage_trace(self):
        """Test LineageTrace creation."""
        from layers import LineageTrace

        trace = LineageTrace(
            query="Test query",
            route="sql",
            sql_run="SELECT 1",
            tables_used=["test_table"],
            schemas_retrieved=["test_schema"],
            documents_retrieved=[],
            cache_hit=False,
            cache_similarity=None,
            execution_time_ms=100.0,
            timestamp="2024-01-01T00:00:00"
        )

        assert trace.query == "Test query"
        assert trace.route == "sql"
        assert len(trace.tables_used) == 1

    def test_lineage_to_dict(self):
        """Test lineage can be converted to dict."""
        from layers import LineageTrace

        trace = LineageTrace(
            query="Test",
            route="sql",
            sql_run=None,
            tables_used=[],
            schemas_retrieved=[],
            documents_retrieved=[],
            cache_hit=False,
            cache_similarity=None,
            execution_time_ms=0,
            timestamp="2024-01-01T00:00:00"
        )

        d = trace.to_dict()
        assert isinstance(d, dict)
        assert "query" in d
        assert "route" in d

    def test_query_response(self):
        """Test QueryResponse dataclass."""
        from layers import QueryResponse, LineageTrace

        trace = LineageTrace(
            query="Test",
            route="sql",
            sql_run=None,
            tables_used=[],
            schemas_retrieved=[],
            documents_retrieved=[],
            cache_hit=False,
            cache_similarity=None,
            execution_time_ms=0,
            timestamp="2024-01-01T00:00:00"
        )

        response = QueryResponse(
            answer="Test answer",
            lineage=trace,
            raw_results=[{"id": 1}]
        )

        assert response.answer == "Test answer"
        assert len(response.raw_results) == 1

    def test_storyteller_sql_rows_override_fallback(self):
        """If SQL rows exist, fallback text should not be returned."""
        from layers.layer6_storyteller import Storyteller

        with patch("layers.layer6_storyteller.GroqClient"):
            storyteller = Storyteller(api_key="dummy")
            with patch.object(storyteller, "_generate_answer", return_value=storyteller.FALLBACK_TEXT):
                answer = storyteller.tell(
                    user_question="Top customers",
                    sql_results=[{"customer_id": "C002"}, {"customer_id": "C001"}, {"customer_id": "C007"}],
                    route="sql",
                )

        assert storyteller.FALLBACK_TEXT not in answer
        assert "C002" in answer

    def test_storyteller_sql_rows_mixed_fallback_text(self):
        """Even if model adds extra text around fallback, deterministic SQL answer should win."""
        from layers.layer6_storyteller import Storyteller

        bad_answer = "I do not have enough information in the current data to answer that. The SQL results are unclear."

        with patch("layers.layer6_storyteller.GroqClient"):
            storyteller = Storyteller(api_key="dummy")
            with patch.object(storyteller, "_generate_answer", return_value=bad_answer):
                answer = storyteller.tell(
                    user_question="Top customers",
                    sql_results=[{"customer_id": "C002"}],
                    route="sql",
                )

        assert storyteller.FALLBACK_TEXT not in answer
        assert "C002" in answer


class TestPipelineIntegration:
    """Integration tests for the main pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        from main_pipeline import AIQuerySystem

        with patch('main_pipeline.SemanticCache'):
            with patch('main_pipeline.IntentRouter'):
                with patch('main_pipeline.TAGRetrieval'):
                    with patch('main_pipeline.MultiAgentSQLEngine'):
                        with patch('main_pipeline.SecureExecutionSandbox'):
                            with patch('main_pipeline.Storyteller'):
                                system = AIQuerySystem(load_sample_schemas=False)
                                assert system is not None

    def test_health_check_returns_dict(self):
        """Test health check returns proper structure."""
        from main_pipeline import AIQuerySystem

        with patch('main_pipeline.SemanticCache') as mock_cache:
            mock_cache_instance = MagicMock()
            mock_cache_instance.is_healthy.return_value = True
            mock_cache.return_value = mock_cache_instance

            with patch('main_pipeline.IntentRouter'):
                with patch('main_pipeline.TAGRetrieval'):
                    with patch('main_pipeline.MultiAgentSQLEngine'):
                        with patch('main_pipeline.SecureExecutionSandbox') as mock_exec:
                            mock_exec_instance = MagicMock()
                            mock_exec_instance.test_connection.return_value = True
                            mock_exec.return_value = mock_exec_instance

                            with patch('main_pipeline.Storyteller'):
                                system = AIQuerySystem(load_sample_schemas=False)
                                health = system.health_check()

                                assert isinstance(health, dict)
                                assert "cache" in health
                                assert "executor" in health


class TestRAGIsolation:
    def _make_system_instance(self):
        from main_pipeline import AIQuerySystem
        return AIQuerySystem.__new__(AIQuerySystem)

    def test_cache_key_changes_by_target_source(self):
        system = self._make_system_instance()
        k1 = system._build_cache_key(
            "q", "rag", "a@x.com", "doc1.pdf", {"session_id": {"$in": ["S1"]}}, ["doc1.pdf"]
        )
        k2 = system._build_cache_key(
            "q", "rag", "a@x.com", "doc2.pdf", {"session_id": {"$in": ["S1"]}}, ["doc2.pdf"]
        )
        assert k1 != k2

    def test_doc_where_filter_includes_user_and_session(self):
        system = self._make_system_instance()
        wf = system._build_doc_where_filter(
            user_email="a@x.com",
            authorized_docs=["doc1.pdf"],
            target_source=None,
            context_filter={"session_id": {"$in": ["Session 1"]}},
        )
        assert wf is not None
        assert "$and" in wf

    def test_doc_where_filter_normalizes_legacy_doc_records(self):
        system = self._make_system_instance()
        wf = system._build_doc_where_filter(
            user_email="a@x.com",
            authorized_docs=[{"file_name": "doc1.pdf"}, "doc2.pdf"],
            target_source=None,
            context_filter={"session_id": {"$in": ["Session 1"]}},
        )
        assert wf is not None
        assert "$and" in wf
        assert {"file_name": {"$in": ["doc1.pdf", "doc2.pdf"]}} in wf["$and"]

    def test_cache_hit_returns_when_results_none(self):
        from main_pipeline import AIQuerySystem

        system = self._make_system_instance()
        system.logger = MagicMock()
        system.cache = MagicMock()
        system.cache.get_exact.return_value = {
            "answer": "Cached RAG answer",
            "similarity": 0.99,
            "metadata": {"route": "rag", "results": None, "docs": [{"id": "d1"}]},
        }
        system.storyteller = MagicMock()
        system.storyteller.create_lineage.return_value = MagicMock()

        response = AIQuerySystem.run_pipeline(system, user_query="same question")

        assert response.answer == "Cached RAG answer"
        assert response.raw_results is None
        assert response.raw_docs == [{"id": "d1"}]

    def test_cache_write_stores_pre_route_and_route_keys(self):
        from main_pipeline import AIQuerySystem

        system = self._make_system_instance()
        system.logger = MagicMock()
        system.config = {"tag": {"top_k_schemas": 5}}
        system.cache = MagicMock()
        system.cache.get_exact.return_value = None
        system.router = MagicMock()
        system.router.route.return_value = {"route": "rag", "schemas": []}
        system.tag = MagicMock()
        system.tag.retrieve_documents.return_value = [{"id": "d1", "content": "doc text", "metadata": {}}]
        system.storyteller = MagicMock()
        system.storyteller.tell.return_value = "Answer from docs"
        system.storyteller.create_lineage.return_value = MagicMock()

        response = AIQuerySystem.run_pipeline(system, user_query="same question")

        assert response.answer == "Answer from docs"
        assert system.cache.set_exact.call_count == 2
        written_keys = [call.args[0] for call in system.cache.set_exact.call_args_list]
        assert len(set(written_keys)) == 2

    def test_pipeline_uses_exact_cache_methods_when_available(self):
        from main_pipeline import AIQuerySystem

        system = self._make_system_instance()
        system.logger = MagicMock()
        system.config = {"tag": {"top_k_schemas": 5}}
        system.cache = MagicMock()
        system.cache.get_exact.return_value = None
        system.router = MagicMock()
        system.router.route.return_value = {"route": "rag", "schemas": []}
        system.tag = MagicMock()
        system.tag.retrieve_documents.return_value = [{"id": "d1", "content": "doc text", "metadata": {}}]
        system.storyteller = MagicMock()
        system.storyteller.tell.return_value = "Answer from docs"
        system.storyteller.create_lineage.return_value = MagicMock()

        response = AIQuerySystem.run_pipeline(system, user_query="same question")

        assert response.answer == "Answer from docs"
        assert system.cache.get_exact.call_count == 1
        assert system.cache.set_exact.call_count == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
