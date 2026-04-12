"""
Main Pipeline - AI Query System
FIX: Replaced all hardcoded OpenAI model names with GROQ_MODELS
FIX: Added graceful fallback when Redis / DB is not available
FIX: Added detailed logging so you can see what each layer is doing
FIX: Cache now only stores results when sql_results is non-empty (prevents caching empty/broken runs)
"""

import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import os
from typing import Dict, Any, Optional, List   # add List here
from dotenv import load_dotenv
load_dotenv()

from layers import (
    SemanticCache,
    IntentRouter,
    TAGRetrieval,
    MultiAgentSQLEngine,
    SecureExecutionSandbox,
    Storyteller,
    LineageTrace,
    QueryResponse,
    create_sample_schemas,
    GROQ_MODELS,
)

logger = logging.getLogger(__name__)


class AIQuerySystem:
    def __init__(self, config_path: Optional[str] = None, load_sample_schemas: bool = True):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self._auto_setup_database()
        self._initialize_layers()
        self._init_document_processor()
        if load_sample_schemas:
            self._load_sample_data()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        config = {}

        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        else:
            for path in ["./config/config.yaml", "../config/config.yaml"]:
                if Path(path).exists():
                    import yaml
                    with open(path, "r") as f:
                        config = yaml.safe_load(f) or {}
                    break

        if not config:
            self.logger.warning("No config.yaml found — using built-in defaults")

        config["db_host"] = os.getenv("DB_HOST", config.get("db_host", "localhost"))
        config["db_port"] = int(os.getenv("DB_PORT", config.get("db_port", 5432)))
        config["db_name"] = os.getenv("DB_NAME", config.get("db_name", "postgres"))
        config["db_user"] = os.getenv("DB_USER", config.get("db_user", "postgres"))
        config["db_password"] = os.getenv("DB_PASSWORD", config.get("db_password", ""))

        config["redis_host"] = os.getenv("REDIS_HOST", config.get("redis_host", "localhost"))
        config["redis_port"] = int(os.getenv("REDIS_PORT", config.get("redis_port", 6379)))

        return config

    def _auto_setup_database(self):
        """Automatically provisions the database on a fresh Docker container."""
        try:
            # MUST HAVE THESE IMPORTS HERE!
            import psycopg2
            from pathlib import Path
            import os
            import time

            # Retry loop to wait for Docker to boot up
            max_retries = 10
            conn = None
            for attempt in range(max_retries):
                try:
                    conn = psycopg2.connect(
                        host=self.config.get("db_host", "127.0.0.1"),
                        port=self.config.get("db_port", 5432),
                        dbname="postgres",
                        user="postgres",
                        password=os.getenv("POSTGRES_PASSWORD", "secret")
                    )
                    break  # Success! Break out of the loop
                except psycopg2.OperationalError:
                    if attempt < max_retries - 1:
                        self.logger.info(f"Database starting up... waiting 2 seconds (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2)
                    else:
                        raise  # Out of retries, throw the error

            conn.autocommit = True
            cursor = conn.cursor()

            cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'customers');")
            if not cursor.fetchone()[0]:
                self.logger.info("Fresh database detected. Running auto-setup...")

                sql_path = Path("setup_db.sql")
                if sql_path.exists():
                    with open(sql_path, "r") as f:
                        sql = f.read()

                    # Inject config values
                    db_name = self.config.get("db_name", "postgres")
                    db_pass = self.config.get("db_password", "1234")

                    sql = sql.replace("yourdatabase", db_name)
                    sql = sql.replace("CREATE ROLE ai_readonly LOGIN;", f"CREATE ROLE ai_readonly LOGIN PASSWORD '{db_pass}';")
                    sql = sql.replace("ALTER ROLE ai_readonly WITH PASSWORD '1234';", f"ALTER ROLE ai_readonly WITH PASSWORD '{db_pass}';")

                    cursor.execute(sql)
                    self.logger.info("Database auto-setup completed successfully!")
                else:
                    self.logger.warning("setup_db.sql not found. Cannot auto-setup database.")

            conn.close()

        except Exception as e:
            # This will now tell us EXACTLY what broke if it fails again
            self.logger.error(f"Auto-setup completely failed: {e}")

    def _initialize_layers(self):
        # Layer 1: Semantic Cache
        cache_config = self.config.get("semantic_cache", {})
        try:
            self.cache = SemanticCache(
                redis_host=self.config.get("redis_host", "localhost"),
                redis_port=self.config.get("redis_port", 6379),
                redis_db=self.config.get("redis_db", 0),
                ttl_seconds=cache_config.get("ttl_seconds", 3600),
                similarity_threshold=cache_config.get("similarity_threshold", 0.92)
            )
            if not self.cache.is_healthy():
                self.logger.warning("Redis not reachable — cache disabled")
                self.cache = None
        except Exception as e:
            self.logger.warning(f"Cache init failed: {e} — cache disabled")
            self.cache = None

        # Layer 2: Intent Router
        router_config = self.config.get("intent_router", {})
        self.router = IntentRouter(
            model=router_config.get("model", GROQ_MODELS["fast"]),
            temperature=router_config.get("temperature", 0.0)
        )

        # Layer 3: TAG Retrieval
        self.tag = TAGRetrieval(
            persist_directory=self.config.get("chroma_persist_dir", "./data/chroma_db")
        )

        # Layer 4: Multi-Agent SQL Engine
        sql_config = self.config.get("multi_agent_sql", {})
        self.sql_engine = MultiAgentSQLEngine(
            planner_model=sql_config.get("planner_model", GROQ_MODELS["powerful"]),
            coder_model=sql_config.get("coder_model", GROQ_MODELS["powerful"]),
            validator_model=sql_config.get("validator_model", GROQ_MODELS["fast"])
        )

        # Layer 5: Secure Execution
        try:
            self.executor = SecureExecutionSandbox(
                db_host=self.config.get("db_host", "localhost"),
                db_port=self.config.get("db_port", 5432),
                db_name=self.config.get("db_name", "postgres"),
                db_user=self.config.get("db_user", "postgres"),
                db_password=self.config.get("db_password", "")
            )
        except Exception as e:
            self.logger.warning(f"DB executor init failed: {e} — SQL execution disabled")
            self.executor = None

        # Layer 6: Storyteller
        storyteller_config = self.config.get("storyteller", {})
        self.storyteller = Storyteller(
            model=storyteller_config.get("model", GROQ_MODELS["powerful"]),
            temperature=storyteller_config.get("temperature", 0.3)
        )

    def _load_sample_data(self):
        try:
            schemas = create_sample_schemas()
            for schema in schemas:
                self.tag.add_schema(schema)
            self.logger.info(f"Loaded {len(schemas)} sample schemas into TAG")

            if hasattr(self.tag, 'add_document'):
                self.tag.add_document(
                    doc_id="policy_001",
                    content="COMPANY REFUND POLICY: All customers are entitled to a full refund within 30 days of purchase. The item must be in its original packaging. Contact support@example.com for processing.",
                    metadata={"source": "employee_handbook"}
                )
                self.logger.info("Loaded sample RAG documents into TAG")

        except Exception as e:
            self.logger.warning(f"Could not load sample data: {e}")

    def _init_document_processor(self):
        """Initialize document processor for file uploads."""
        try:
            from document_processor import create_document_processor
            self.doc_processor = create_document_processor(
                tag=self.tag,
                executor=self.executor,
                config=self.config
            )
            self.logger.info("Document processor initialized")
        except Exception as e:
            self.logger.warning(f"Document processor init failed: {e}")
            self.doc_processor = None

    def upload_file(self, file_path: str, original_file_name: Optional[str] = None) -> Dict[str, Any]:
        """Upload and process a single file (CSV/Excel/JSON → SQL, PDF/TXT/DOCX → RAG)."""
        if not self.doc_processor:
            return {"success": False, "message": "Document processor not initialized"}
        return self.doc_processor.process(file_path, original_file_name=original_file_name)

    def upload_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Upload multiple files at once."""
        if not self.doc_processor:
            return [{"success": False, "message": "Document processor not initialized"}]
        return self.doc_processor.process_many(file_paths)

    def list_uploads(self) -> Dict[str, Any]:
        """Show all currently loaded schemas and RAG documents."""
        if not self.doc_processor:
            return {"schemas": [], "documents": []}
        return {
            "schemas":   self.doc_processor.list_loaded_schemas(),
            "documents": self.doc_processor.list_loaded_documents()
        }

# Change the signature to include target_source
    def run_pipeline(
        self,
        user_query: str,
        context_filter: Optional[Dict[str, Any]] = None,
        authorized_docs: Optional[list] = None,
        target_source: Optional[str] = None
        ) -> QueryResponse:
        self.logger.info(f"Query: {user_query}")
        start_time = time.time()

        # Step 1: Routing
        route_result = self.router.route(user_query)
        route = route_result["route"]

        # ==========================================
        # NEW: SMART ROUTING OVERRIDE based on File Type
        # ==========================================
        if target_source:
            structured_exts = ['.csv', '.xlsx', '.xls', '.json']
            unstructured_exts = ['.pdf', '.txt', '.docx', '.md']

            target_lower = target_source.lower()
            if any(target_lower.endswith(ext) for ext in structured_exts):
                if route == "rag":
                    route = "sql"  # Force SQL for spreadsheets
                    self.logger.info(f"[ROUTER OVERRIDE] Forced SQL route for structured file: {target_source}")
            elif any(target_lower.endswith(ext) for ext in unstructured_exts):
                if route == "sql":
                    route = "rag"  # Force RAG for documents
                    self.logger.info(f"[ROUTER OVERRIDE] Forced RAG route for unstructured file: {target_source}")
        # ==========================================

        # Step 2: Semantic Cache Check
        cache_hit = False
        cached_result = self.cache.get(user_query)
        if cached_result:
            self.logger.info("[CACHE] Hit found")
            cache_hit = True
            return QueryResponse(
                answer=cached_result["answer"],
                lineage=LineageTrace(**cached_result["lineage"])
            )

        # Step 3: Retrieve schemas / documents
        schemas, docs, schema_context = [], [], ""

        # --- SQL RETRIEVAL ---
        if route in ["sql", "both"]:
            # Inject the target source into the search query so ChromaDB finds the matching table schema!
            schema_search_query = f"{user_query} {target_source}" if target_source else user_query
            schemas = self.tag.retrieve_schemas(schema_search_query, top_k=3)
            schema_context = "\n\n".join([s.to_document() for s in schemas])
            self.logger.info(f"[TAG] Retrieved schemas: {[s.table_name for s in schemas]}")

        # --- RAG RETRIEVAL ---
        if route in ["rag", "both"]:
            conditions = []

            if target_source:
                conditions.append({"source": target_source})

            if authorized_docs is not None:
                if len(authorized_docs) == 0:
                    conditions.append({"file_name": "NO_ACCESS_NULL_FILE"})
                elif len(authorized_docs) == 1:
                    conditions.append({"file_name": authorized_docs[0]})
                else:
                    conditions.append({"file_name": {"$in": authorized_docs}})

            if context_filter:
                for key, val in context_filter.items():
                    conditions.append({key: val})

            if len(conditions) == 1:
                where_filter = conditions[0]
            elif len(conditions) > 1:
                where_filter = {"$and": conditions}
            else:
                where_filter = None

            docs = self.tag.retrieve_documents(user_query, top_k=5, where_filter=where_filter)
            self.logger.info(f"[TAG] Retrieved {len(docs)} documents with filter: {where_filter}")

        # ... (Leave Step 4: SQL Execution and the rest of the function exactly as it is) ...
        sql_results, sql_query, tables_used = None, None, []

        if route in ["sql", "both"] and schema_context:
            if not self.executor:
                self.logger.warning("[SQL] Executor offline — skipping SQL execution. Check DB config.")
            else:
                self.logger.info("[SQL ENGINE] Generating SQL via multi-agent pipeline...")
                sql_result = self.sql_engine.execute(user_query, schema_context)

                if sql_result.success:
                    sql_query = sql_result.query
                    tables_used = sql_result.tables_used
                    self.logger.info(f"[SQL ENGINE] Generated SQL:\n{sql_query}")
                    self.logger.info(f"[SQL ENGINE] Tables used: {tables_used}")

                    try:
                        db_result = self.executor.execute(sql_query)
                        if db_result.success:
                            sql_results = db_result.rows
                            self.logger.info(f"[EXECUTOR] Got {len(sql_results)} rows in {db_result.execution_time_ms:.1f}ms")
                            if sql_results:
                                self.logger.info(f"[EXECUTOR] Sample row: {sql_results[0]}")
                        else:
                            self.logger.error(f"[EXECUTOR] DB error: {db_result.error}")
                    except Exception as e:
                        self.logger.error(f"[EXECUTOR] Exception: {e}")
                else:
                    self.logger.warning(f"[SQL ENGINE] Validation failed: {sql_result.validation_errors}")

        # Step 6: Generate natural language answer
        self.logger.info(f"[STORYTELLER] Generating answer (sql_results={'yes' if sql_results else 'none'}, docs={len(docs)})...")
        answer = self.storyteller.tell(
            user_question=user_query,
            sql_results=sql_results,
            doc_context=docs,
            route=route
        )
        self.logger.info(f"[STORYTELLER] Answer: {answer[:100]}...")

        # Step 7: Lineage
        total_ms = (time.time() - start_time) * 1000
        lineage = self.storyteller.create_lineage(
            query=user_query, route=route, sql_query=sql_query,
            tables_used=tables_used,
            schemas_retrieved=[s.table_name for s in schemas],
            documents_retrieved=[d.get("id", "") for d in docs],
            cache_hit=False,
            execution_time_ms=total_ms
        )

        # FIX: Only cache when we have actual results
        # ====================================================================================
        # OLD CACHE SAVE LOGIC (PRESERVED):
        # if self.cache and sql_results:
        #     try:
        #         self.cache.set(user_query, answer, metadata={"route": route, "results": sql_results})
        #         self.logger.info(f"[CACHE] Stored answer with {len(sql_results)} rows")
        #     except Exception as e:
        #         self.logger.warning(f"Cache write failed: {e}")
        # ====================================================================================

        # NEW FIX: Save against isolated Cache Key and support RAG docs caching
        if self.cache and (sql_results or docs):
            try:
                cache_key = user_query
                if context_filter:
                    cache_key = f"{user_query}__CTX__{str(context_filter)}"
                self.cache.set(cache_key, answer, metadata={"route": route, "results": sql_results})
                self.logger.info(f"[CACHE] Stored locally isolated cache entry.")
            except Exception as e:
                self.logger.warning(f"Cache write failed: {e}")

        self.storyteller.log_lineage(lineage)
        return QueryResponse(answer=answer, lineage=lineage, raw_results=sql_results, raw_docs=docs)

    def clear_cache(self) -> int:
        """Clear all cache entries. Useful after fixing bugs."""
        if self.cache:
            return self.cache.clear()
        return 0

    def get_available_sources(self) -> list:
        """Get a list of unique document sources from the RAG database."""
        if not self.tag or not hasattr(self.tag, 'docs_collection'):
            return []
        try:
            # Fetch all metadata from ChromaDB
            result = self.tag.docs_collection.get(include=["metadatas"])
            sources = set()
            for meta in result.get("metadatas", []):
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return sorted(list(sources))
        except Exception as e:
            self.logger.error(f"Failed to get sources: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        return {
            "cache":       self.cache.is_healthy() if self.cache else False,
            "router":      True,
            "tag":         True,
            "sql_engine":  True,
            "executor":    self.executor.test_connection() if self.executor else False,
            "storyteller": True
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "cache_stats": self.cache.get_stats() if self.cache else {"total_entries": 0},
            "tag_collections": {
                "schemas":   self.tag.schema_collection.count(),
                "documents": self.tag.docs_collection.count()
            },
            "recent_lineage": self.storyteller.get_lineage_logs(limit=10)
        }


def run_demo():
    print("=" * 60)
    print("AI Query System - Demo")
    print("=" * 60)

    system = AIQuerySystem(load_sample_schemas=True)

    # FIX: Always clear cache at demo start so stale empty results don't block
    cleared = system.clear_cache()
    if cleared:
        print(f"\nCleared {cleared} stale cache entries")

    print("\nHealth Check:")
    for component, status in system.health_check().items():
        print(f"  {component}: {'OK' if status else 'OFFLINE'}")

    demo_queries = [
        "How many customers do we have?",
        "What is the total revenue?",
        "Show me recent orders",
        "What is the company refund policy?"
    ]

    print("\n" + "=" * 60)
    for query in demo_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        try:
            response = system.run_pipeline(query)
            print(f"Answer: {response.answer}")
            print(f"Route:  {response.lineage.route}")
            print(f"Cache:  {response.lineage.cache_hit}")
            print(f"Tables: {response.lineage.tables_used}")
            print(f"SQL:    {response.lineage.sql_run}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s"
    )
    run_demo()
