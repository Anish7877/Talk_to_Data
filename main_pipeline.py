# """
# Main Pipeline - AI Query System
# FIX: Replaced all hardcoded OpenAI model names with GROQ_MODELS
# FIX: Added graceful fallback when Redis / DB is not available
# FIX: Added detailed logging so you can see what each layer is doing
# FIX: Cache now only stores results when sql_results is non-empty (prevents caching empty/broken runs)
# """

# import time
# import logging
# from typing import Dict, Any, Optional
# from pathlib import Path
# import os
# from typing import Dict, Any, Optional, List   # add List here
# from dotenv import load_dotenv
# load_dotenv()

# from layers import (
#     SemanticCache,
#     IntentRouter,
#     TAGRetrieval,
#     MultiAgentSQLEngine,
#     SecureExecutionSandbox,
#     Storyteller,
#     LineageTrace,
#     QueryResponse,
#     create_sample_schemas,
#     GROQ_MODELS,
# )

# logger = logging.getLogger(__name__)


# class AIQuerySystem:
#     def __init__(self, config_path: Optional[str] = None, load_sample_schemas: bool = True):
#         self.logger = logging.getLogger(__name__)
#         self.config = self._load_config(config_path)
#         self._auto_setup_database()
#         self._initialize_layers()
#         self._init_document_processor()
#         if load_sample_schemas:
#             self._load_sample_data()

#     def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
#         config = {}

#         if config_path and Path(config_path).exists():
#             import yaml
#             with open(config_path, "r") as f:
#                 config = yaml.safe_load(f) or {}
#         else:
#             for path in ["./config/config.yaml", "../config/config.yaml"]:
#                 if Path(path).exists():
#                     import yaml
#                     with open(path, "r") as f:
#                         config = yaml.safe_load(f) or {}
#                     break

#         if not config:
#             self.logger.warning("No config.yaml found — using built-in defaults")

#         config["db_host"] = os.getenv("DB_HOST", config.get("db_host", "localhost"))
#         config["db_port"] = int(os.getenv("DB_PORT", config.get("db_port", 5432)))
#         config["db_name"] = os.getenv("DB_NAME", config.get("db_name", "postgres"))
#         config["db_user"] = os.getenv("DB_USER", config.get("db_user", "postgres"))
#         config["db_password"] = os.getenv("DB_PASSWORD", config.get("db_password", ""))

#         config["redis_host"] = os.getenv("REDIS_HOST", config.get("redis_host", "localhost"))
#         config["redis_port"] = int(os.getenv("REDIS_PORT", config.get("redis_port", 6379)))

#         return config

#     def _auto_setup_database(self):
#         """Automatically provisions the database on a fresh Docker container."""
#         try:
#             # MUST HAVE THESE IMPORTS HERE!
#             import psycopg2
#             from pathlib import Path
#             import time

#             # Retry loop to wait for Docker to boot up
#             max_retries = 10
#             conn = None
#             for attempt in range(max_retries):
#                 try:
#                     conn = psycopg2.connect(
#                         host=self.config.get("db_host", "127.0.0.1"),
#                         port=self.config.get("db_port", 5432),
#                         dbname="postgres",
#                         user="postgres",
#                         password=os.getenv("POSTGRES_PASSWORD", "secret")
#                     )
#                     break  # Success! Break out of the loop
#                 except psycopg2.OperationalError:
#                     if attempt < max_retries - 1:
#                         self.logger.info(f"Database starting up... waiting 2 seconds (attempt {attempt + 1}/{max_retries})")
#                         time.sleep(2)
#                     else:
#                         raise  # Out of retries, throw the error

#             conn.autocommit = True
#             cursor = conn.cursor()

#             cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'customers');")
#             if not cursor.fetchone()[0]:
#                 self.logger.info("Fresh database detected. Running auto-setup...")

#                 sql_path = Path("setup_db.sql")
#                 if sql_path.exists():
#                     with open(sql_path, "r") as f:
#                         sql = f.read()

#                     # Inject config values
#                     db_name = self.config.get("db_name", "postgres")
#                     db_pass = self.config.get("db_password", "1234")

#                     sql = sql.replace("yourdatabase", db_name)
#                     sql = sql.replace("CREATE ROLE ai_readonly LOGIN;", f"CREATE ROLE ai_readonly LOGIN PASSWORD '{db_pass}';")
#                     sql = sql.replace("ALTER ROLE ai_readonly WITH PASSWORD '1234';", f"ALTER ROLE ai_readonly WITH PASSWORD '{db_pass}';")

#                     cursor.execute(sql)
#                     self.logger.info("Database auto-setup completed successfully!")
#                 else:
#                     self.logger.warning("setup_db.sql not found. Cannot auto-setup database.")

#             conn.close()

#         except Exception as e:
#             # This will now tell us EXACTLY what broke if it fails again
#             self.logger.error(f"Auto-setup completely failed: {e}")

#     def _initialize_layers(self):
#         # Layer 1: Semantic Cache
#         cache_config = self.config.get("semantic_cache", {})
#         try:
#             self.cache = SemanticCache(
#                 redis_host=self.config.get("redis_host", "localhost"),
#                 redis_port=self.config.get("redis_port", 6379),
#                 redis_db=self.config.get("redis_db", 0),
#                 ttl_seconds=cache_config.get("ttl_seconds", 3600),
#                 similarity_threshold=cache_config.get("similarity_threshold", 0.92)
#             )
#             if not self.cache.is_healthy():
#                 self.logger.warning("Redis not reachable — cache disabled")
#                 self.cache = None
#         except Exception as e:
#             self.logger.warning(f"Cache init failed: {e} — cache disabled")
#             self.cache = None

#         # Layer 2: Intent Router
#         router_config = self.config.get("intent_router", {})
#         self.router = IntentRouter(
#             model=router_config.get("model", GROQ_MODELS["fast"]),
#             temperature=router_config.get("temperature", 0.0)
#         )

#         # Layer 3: TAG Retrieval
#         self.tag = TAGRetrieval(
#             persist_directory=self.config.get("chroma_persist_dir", "./data/chroma_db")
#         )

#         # Layer 4: Multi-Agent SQL Engine
#         sql_config = self.config.get("multi_agent_sql", {})
#         self.sql_engine = MultiAgentSQLEngine(
#             planner_model=sql_config.get("planner_model", GROQ_MODELS["powerful"]),
#             coder_model=sql_config.get("coder_model", GROQ_MODELS["powerful"]),
#             validator_model=sql_config.get("validator_model", GROQ_MODELS["fast"])
#         )

#         # Layer 5: Secure Execution
#         try:
#             self.executor = SecureExecutionSandbox(
#                 db_host=self.config.get("db_host", "localhost"),
#                 db_port=self.config.get("db_port", 5432),
#                 db_name=self.config.get("db_name", "postgres"),
#                 db_user=self.config.get("db_user", "postgres"),
#                 db_password=self.config.get("db_password", "")
#             )
#         except Exception as e:
#             self.logger.warning(f"DB executor init failed: {e} — SQL execution disabled")
#             self.executor = None

#         # Layer 6: Storyteller
#         storyteller_config = self.config.get("storyteller", {})
#         self.storyteller = Storyteller(
#             model=storyteller_config.get("model", GROQ_MODELS["powerful"]),
#             temperature=storyteller_config.get("temperature", 0.3)
#         )

#     def _load_sample_data(self):
#         try:
#             schemas = create_sample_schemas()
#             for schema in schemas:
#                 self.tag.add_schema(schema)
#             self.logger.info(f"Loaded {len(schemas)} sample schemas into TAG")

#             if hasattr(self.tag, 'add_document'):
#                 self.tag.add_document(
#                     doc_id="policy_001",
#                     content="COMPANY REFUND POLICY: All customers are entitled to a full refund within 30 days of purchase. The item must be in its original packaging. Contact support@example.com for processing.",
#                     metadata={"source": "employee_handbook"}
#                 )
#                 self.logger.info("Loaded sample RAG documents into TAG")

#         except Exception as e:
#             self.logger.warning(f"Could not load sample data: {e}")

#     def _init_document_processor(self):
#         """Initialize document processor for file uploads."""
#         try:
#             from document_processor import create_document_processor
#             self.doc_processor = create_document_processor(
#                 tag=self.tag,
#                 executor=self.executor,
#                 config=self.config
#             )
#             self.logger.info("Document processor initialized")
#         except Exception as e:
#             self.logger.warning(f"Document processor init failed: {e}")
#             self.doc_processor = None

#     # def upload_file(self, file_path: str, original_file_name: Optional[str] = None) -> Dict[str, Any]:
#     #     """Upload and process a single file (CSV/Excel/JSON → SQL, PDF/TXT/DOCX → RAG)."""
#     #     if not self.doc_processor:
#     #         return {"success": False, "message": "Document processor not initialized"}
#     #     return self.doc_processor.process(file_path, original_file_name=original_file_name)

#     def upload_file(
#     self,
#     file_path: str,
#     original_file_name: Optional[str] = None,
#     user_email: Optional[str] = None,
#     session_id: Optional[str] = None,
#     upload_ts: Optional[str] = None,
# ) -> Dict[str, Any]:
#         """Upload and process a single file (CSV/Excel/JSON -> SQL, PDF/TXT/DOCX -> RAG)."""
#         if not self.doc_processor:
#             return {"success": False, "message": "Document processor not initialized"}

#         return self.doc_processor.process(
#             file_path,
#             original_file_name=original_file_name,
#             user_email=user_email,
#             session_id=session_id,
#             upload_ts=upload_ts,
#             )

#     def upload_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
#         """Upload multiple files at once."""
#         if not self.doc_processor:
#             return [{"success": False, "message": "Document processor not initialized"}]
#         return self.doc_processor.process_many(file_paths)

#     def list_uploads(self) -> Dict[str, Any]:
#         """Show all currently loaded schemas and RAG documents."""
#         if not self.doc_processor:
#             return {"schemas": [], "documents": []}
#         return {
#             "schemas":   self.doc_processor.list_loaded_schemas(),
#             "documents": self.doc_processor.list_loaded_documents()
#         }


#     def _build_doc_where_filter(
#     self,
#     user_email: Optional[str],
#     authorized_docs: Optional[List[str]],
#     target_source: Optional[str],
#     context_filter: Optional[Dict[str, Any]],
# ) -> Optional[Dict[str, Any]]:
#         conditions = []

#         if user_email:
#             conditions.append({"user_email": user_email})

#         if authorized_docs is not None:
#             if len(authorized_docs) == 0:
#                 conditions.append({"file_name": "__NO_AUTH__"})
#             elif len(authorized_docs) == 1:
#                 conditions.append({"file_name": str(authorized_docs[0])})
#             else:
#                 conditions.append({"file_name": {"$in": [str(d) for d in authorized_docs]}})

#         if target_source:
#             conditions.append({"file_name": target_source})

#         if context_filter:
#             for key, val in context_filter.items():
#                 conditions.append({key: val})

#         if not conditions:
#             return None
#         if len(conditions) == 1:
#             return conditions[0]
#         return {"$and": conditions}



#     def _build_cache_key(
#     self,
#     user_query: str,
#     route: str,
#     user_email: Optional[str],
#     target_source: Optional[str],
#     context_filter: Optional[Dict[str, Any]],
#     authorized_docs: Optional[List[str]],
# ) -> str:
#     import hashlib
#     import json

#     payload = { 
#         "q": user_query,
#         "route": route,
#         "user_email": user_email or "",
#         "target_source": target_source or "",
#         "context_filter": context_filter or {},
#         "authorized_docs": sorted([str(d) for d in (authorized_docs or [])]),
#     }
#     raw = json.dumps(payload, sort_keys=True)
#     return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# # Change the signature to include target_source
#     def run_pipeline(
#         self,
#         user_query: str,
#         context_filter: Optional[Dict[str, Any]] = None,
#         authorized_docs: Optional[List[str]] = None,
#         target_source: Optional[str] = None,
#          user_email: Optional[str] = None,
#     ) -> QueryResponse:
#         start_time = time.time()
#         self.logger.info(f"Query: {user_query}")

#         # Step 1: Semantic Cache Check
#         # if self.cache:
#         #     try:
#         #         import json
#         #         cache_key = user_query
#         #         if context_filter:
#         #             ctx_str = json.dumps(context_filter, sort_keys=True)
#         #             cache_key = f"{user_query}__CTX__{ctx_str}"

#         #         cached = self.cache.get(cache_key)
#         #         if cached:
#         #             cached_results = cached.get("metadata", {}).get("results")
#         #             if cached_results:
#         #                 self.logger.info(f"[CACHE HIT] similarity={cached.get('similarity', 0):.3f}")
#         #                 lineage = self.storyteller.create_lineage(
#         #                     query=user_query, route="cache",
#         #                     cache_hit=True, cache_similarity=cached.get("similarity"),
#         #                     execution_time_ms=0
#         #                 )
#         #                 return QueryResponse(
#         #                     answer=cached["answer"],
#         #                     lineage=lineage,
#         #                     raw_results=cached_results
#         #                 )
#         #     except Exception as e:
#         #         self.logger.warning(f"Cache lookup failed: {e}")




#         # Step 1: Semantic Cache Check
#     if self.cache:
#         try:
#             # route not finalized yet; use "pre_route" for lookup stage
#             pre_route_cache_key = self._build_cache_key(
#                 user_query=user_query,
#                 route="pre_route",
#                 user_email=user_email,
#                 target_source=target_source,
#                 context_filter=context_filter,
#                 authorized_docs=authorized_docs,
#             )
#             cached = self.cache.get(pre_route_cache_key)
#             if cached:
#             cached_results = cached.get("metadata", {}).get("results")
#             if cached_results is not None:
#                 self.logger.info(f"[CACHE HIT] similarity={cached.get('similarity', 0):.3f}")
#                 lineage = self.storyteller.create_lineage(
#                     query=user_query,
#                     route="cache",
#                     cache_hit=True,
#                     cache_similarity=cached.get("similarity"),
#                     execution_time_ms=0,
#                 )
#                 return QueryResponse(
#                     answer=cached["answer"],
#                     lineage=lineage,
#                     raw_results=cached_results,
#                 )
#         except Exception as e:
#             self.logger.warning(f"Cache lookup failed: {e}")
        





#         # Step 2: Route query & Smart Override
#         import re
#         routing = self.router.route(user_query)
#         route = routing.get("route", "sql")
#         inferred_schemas = routing.get("schemas", [])

#         if target_source:
#             structured_exts = ['.csv', '.xlsx', '.xls', '.json']
#             unstructured_exts = ['.pdf', '.txt', '.docx', '.md']

#             target_lower = target_source.lower()
#             if any(target_lower.endswith(ext) for ext in structured_exts):
#                 if route == "rag":
#                     route = "sql"
#                     self.logger.info(f"[ROUTER OVERRIDE] Forced SQL route for structured file: {target_source}")
#             elif any(target_lower.endswith(ext) for ext in unstructured_exts):
#                 if route == "sql":
#                     route = "rag"
#                     self.logger.info(f"[ROUTER OVERRIDE] Forced RAG route for unstructured file: {target_source}")

#         self.logger.info(f"[ROUTER] route={route} | inferred_schemas={inferred_schemas}")

#         # Setup Search Term (Handling @mentions)
#         mentions = re.findall(r"@([a-zA-Z0-9_.\-]+)", user_query)
#         combined_hints = []
#         for m in mentions:
#             if m not in combined_hints: combined_hints.append(m)
#         for sc in inferred_schemas:
#             if sc not in combined_hints: combined_hints.append(sc)

#         search_term = user_query
#         if combined_hints:
#             search_term = search_term + " " + " ".join(combined_hints)

#         # Step 3: Retrieve schemas / documents
#         schemas, docs, schema_context = [], [], ""

#         # --- SQL SCHEMA RETRIEVAL (This was missing!) ---
#         if route in ["sql", "both"]:
#             schema_where = None
#             if target_source:
#                 from pathlib import Path
#                 structured_exts = ['.csv', '.xlsx', '.xls', '.json']
#                 if any(target_source.lower().endswith(ext) for ext in structured_exts):
#                     target_table = Path(target_source).stem.lower().replace(" ", "_").replace("-", "_")
#                     schema_where = {"table_name": target_table}

#             schemas = self.tag.retrieve_schemas(search_term, top_k=2, where_filter=schema_where)
#             schema_context = "\n\n".join([s.to_document()[:800] for s in schemas])
#             self.logger.info(f"[TAG] Retrieved schemas: {[s.table_name for s in schemas]} with filter: {schema_where}")

#         # --- RAG DOCUMENT RETRIEVAL ---
#         # if route in ["rag", "both"]:
#         #     conditions = []
#         #     if target_source:
#         #         # FIX: Match the exact metadata key used by document_processor.py
#         #         conditions.append({"file_name": target_source})

#         #     if authorized_docs is not None:
#         #         if len(authorized_docs) == 0:
#         #             conditions.append({"file_name": "NO_ACCESS_NULL_FILE"})
#         #         elif len(authorized_docs) == 1:
#         #             conditions.append({"file_name": authorized_docs[0]})
#         #         else:
#         #             conditions.append({"file_name": {"$in": authorized_docs}})

#         #     if context_filter:
#         #         for key, val in context_filter.items():
#         #             conditions.append({key: val})

#         #     if len(conditions) == 1:
#         #         where_filter = conditions[0]
#         #     elif len(conditions) > 1:
#         #         where_filter = {"$and": conditions}
#         #     else:
#         #         where_filter = None

#         #     docs = self.tag.retrieve_documents(search_term, top_k=5, where_filter=where_filter)
#         #     self.logger.info(f"[TAG] Retrieved {len(docs)} documents with filter: {where_filter}")


#         if route in ["rag", "both"]:
#             where_filter = self._build_doc_where_filter(
#                 user_email=user_email,
#                 authorized_docs=authorized_docs,
#                 target_source=target_source,
#                 context_filter=context_filter,
#             )

#         docs = self.tag.retrieve_documents(search_term, top_k=12, where_filter=where_filter)
#         docs = docs[:5]  # final context set
#         self.logger.info(f"[TAG] Retrieved {len(docs)} documents with filter: {where_filter}")

#         # Steps 4 & 5: Generate SQL and execute (Self-Healing Loop)
#         sql_results, sql_query, tables_used = None, None, []

#         if route in ["sql", "both"] and schema_context:
#             if not self.executor:
#                 self.logger.warning("[SQL] Executor offline — skipping SQL execution. Check DB config.")
#             else:
#                 self.logger.info("[SQL ENGINE] Generating SQL via multi-agent pipeline...")

#                 max_retries = 3
#                 feedback_history = ""

#                 for attempt in range(max_retries):
#                     self.logger.info(f"[SQL ENGINE] Attempt {attempt + 1}/{max_retries}...")

#                     enhanced_query = user_query
#                     if feedback_history:
#                         enhanced_query += f"\n\nDO NOT repeat previous mistakes. Previous attempts failed with:\n{feedback_history}"

#                     sql_result = self.sql_engine.execute(enhanced_query, schema_context)

#                     if sql_result.success:
#                         sql_query = sql_result.query
#                         tables_used = sql_result.tables_used
#                         self.logger.info(f"[SQL ENGINE] Generated SQL:\n{sql_query}")

#                         try:
#                             db_result = self.executor.execute(sql_query)
#                             if db_result.success:
#                                 sql_results = db_result.rows
#                                 self.logger.info(f"[EXECUTOR] Got {len(sql_results)} rows in {db_result.execution_time_ms:.1f}ms")
#                                 break # Success!
#                             else:
#                                 self.logger.warning(f"[EXECUTOR] DB error: {db_result.error}. Retrying...")
#                                 feedback_history += f"- Query: {sql_query}\n- Error: {db_result.error}\n"
#                         except Exception as e:
#                             self.logger.error(f"[EXECUTOR] Exception: {e}")
#                             break
#                     else:
#                         self.logger.warning(f"[SQL ENGINE] Validation failed: {sql_result.validation_errors}")
#                         feedback_history += f"- Validation Error: {sql_result.validation_errors}\n"

#                 if not sql_results and feedback_history:
#                     self.logger.error("[SQL ENGINE] All retry attempts failed.")



#         # Step 6: Generate natural language answer

#         # --- 🕵️ HACKATHON DEBUG: WHAT IS THE AI ACTUALLY SEEING? ---
#         if route in ["rag", "both"]:
#             print("\n" + "!" * 50)
#             print("🕵️ X-RAY VISION: RAG DOCUMENTS RETRIEVED")
#             print("!" * 50)
#             if not docs:
#                 print("❌ ERROR: 0 chunks retrieved! The ChromaDB filter failed or the PDF is empty.")
#             else:
#                 for i, d in enumerate(docs):
#                     print(f"\n--- CHUNK {i+1} (Score: {d.get('distance', 'N/A')}) ---")
#                     # Print the first 400 characters of what pypdf actually extracted
#                     print(d.get('content', '')[:400])
#             print("!" * 50 + "\n")
#         # -----------------------------------------------------------

#         # Step 6: Generate natural language answer
#         self.logger.info(f"[STORYTELLER] Generating answer (sql_results={'yes' if sql_results else 'none'}, docs={len(docs)})...")
#         answer = self.storyteller.tell(
#             user_question=user_query,
#             sql_results=sql_results,
#             doc_context=docs,
#             route=route
#         )
#         self.logger.info(f"[STORYTELLER] Answer: {answer[:100]}...")

#         # Step 7: Lineage
#         total_ms = (time.time() - start_time) * 1000
#         lineage = self.storyteller.create_lineage(
#             query=user_query, route=route, sql_query=sql_query,
#             tables_used=tables_used,
#             schemas_retrieved=[s.table_name for s in schemas],
#             documents_retrieved=[d.get("id", "") for d in docs],
#             cache_hit=False,
#             execution_time_ms=total_ms
#         )

#         # if self.cache and (sql_results or docs):
#         #     try:
#         #         import json
#         #         cache_key = user_query
#         #         if context_filter:
#         #             ctx_str = json.dumps(context_filter, sort_keys=True)
#         #             cache_key = f"{user_query}__CTX__{ctx_str}"
#         #         self.cache.set(cache_key, answer, metadata={"route": route, "results": sql_results})
#         #         self.logger.info(f"[CACHE] Stored locally isolated cache entry.")
#         #     except Exception as e:
#         #         self.logger.warning(f"Cache write failed: {e}")

#         if self.cache and (sql_results or docs):
#             try:
#                 cache_key = self._build_cache_key(
#                     user_query=user_query,
#                     route=route,
#                     user_email=user_email,
#                     target_source=target_source,
#                     context_filter=context_filter,
#                     authorized_docs=authorized_docs,
#                 )
#                 self.cache.set(
#                     cache_key,
#                     answer,
#                     metadata={"route": route, "results": sql_results, "docs_count": len(docs)},
#                 )
#                 self.logger.info("[CACHE] Stored isolated cache entry.")

#             except Exception as e:
#                 self.logger.warning(f"Cache write failed: {e}")

#         self.storyteller.log_lineage(lineage)
#         return QueryResponse(answer=answer, lineage=lineage, raw_results=sql_results, raw_docs=docs)

#     def clear_cache(self) -> int:
#         """Clear all cache entries. Useful after fixing bugs."""
#         if self.cache:
#             return self.cache.clear()
#         return 0

#     def get_available_sources(self) -> list:
#         """Get a list of unique document sources from the RAG database."""
#         if not self.tag or not hasattr(self.tag, 'docs_collection'):
#             return []
#         try:
#             # Fetch all metadata from ChromaDB
#             result = self.tag.docs_collection.get(include=["metadatas"])
#             sources = set()
#             # for meta in result.get("metadatas", []):
#             #     if meta and "source" in meta:
#             #         sources.add(meta["source"])
#             for meta in result.get("metadatas", []):
#                 if not meta:
#                     continue
#                 if "file_name" in meta and meta["file_name"]:
#                     sources.add(meta["file_name"])
#                 elif "source" in meta and meta["source"]:
#                     sources.add(meta["source"])
#             return sorted(list(sources))
#         except Exception as e:
#             self.logger.error(f"Failed to get sources: {e}")
#             return []

#     def health_check(self) -> Dict[str, Any]:
#         return {
#             "cache":       self.cache.is_healthy() if self.cache else False,
#             "router":      True,
#             "tag":         True,
#             "sql_engine":  True,
#             "executor":    self.executor.test_connection() if self.executor else False,
#             "storyteller": True
#         }

#     def get_stats(self) -> Dict[str, Any]:
#         return {
#             "cache_stats": self.cache.get_stats() if self.cache else {"total_entries": 0},
#             "tag_collections": {
#                 "schemas":   self.tag.schema_collection.count(),
#                 "documents": self.tag.docs_collection.count()
#             },
#             "recent_lineage": self.storyteller.get_lineage_logs(limit=10)
#         }


# def run_demo():
#     import os

#     print("=" * 60)
#     print("AI Query System - STRICT TARGETING DEMO")
#     print("=" * 60)

#     system = AIQuerySystem(load_sample_schemas=True)
#     cleared = system.clear_cache()
#     if cleared:
#         print(f"\nCleared {cleared} stale cache entries")

#     # ==========================================
#     # STEP 1: UPLOAD FILES
#     # ==========================================
#     print("\n" + "=" * 60)
#     print("STEP 1: Upload Your Documents")
#     print("=" * 60)

#     active_files = []

#     unstructured_file = input("Path to an unstructured file (e.g., ./docs/manual.pdf) [Enter to skip]: ").strip()
#     if unstructured_file and os.path.exists(unstructured_file):
#         res = system.upload_file(unstructured_file)
#         if res.get('success'):
#             active_files.append(os.path.basename(unstructured_file))
#             print(f" -> Success! Added {os.path.basename(unstructured_file)}")
#         else:
#             print(f" -> Failed: {res.get('message')}")

#     structured_file = input("Path to a structured file (e.g., ./data/inventory.csv) [Enter to skip]: ").strip()
#     if structured_file and os.path.exists(structured_file):
#         res = system.upload_file(structured_file)
#         if res.get('success'):
#             active_files.append(os.path.basename(structured_file))
#             print(f" -> Success! Added {os.path.basename(structured_file)}")
#         else:
#             print(f" -> Failed: {res.get('message')}")

#     if not active_files:
#         print("\nNo files were successfully uploaded. Exiting demo.")
#         return

#     # ==========================================
#     # STEP 2: FORCED INTERACTIVE LOOP
#     # ==========================================
#     print("\n" + "=" * 60)
#     print("STEP 2: Strict Querying")
#     print("=" * 60)

#     while True:
#         print("\nAVAILABLE FILES:")
#         for i, f in enumerate(active_files, 1):
#             print(f"[{i}] {f}")
#         print(f"[{len(active_files) + 1}] Exit Demo")

#         # 1. FORCE THE FILE SELECTION FIRST
#         file_choice = input("\nWhich file do you want to query? (Enter the number): ").strip()

#         if file_choice == str(len(active_files) + 1):
#             print("Exiting demo...")
#             break

#         try:
#             choice_idx = int(file_choice) - 1
#             if choice_idx < 0 or choice_idx >= len(active_files):
#                 print("Invalid number. You MUST select a valid file from the list.")
#                 continue
#             target_file = active_files[choice_idx]
#         except ValueError:
#             print("Please enter a valid number.")
#             continue

#         # 2. ONLY ASK FOR QUERY AFTER FILE IS SECURED
#         query = input(f"\nEnter your query specifically for '{target_file}': ").strip()
#         if not query:
#             continue

#         # 3. EXECUTE WITH GUARANTEED CONTEXT
#         print(f"\n--- [Locking pipeline context strictly to: {target_file}] ---")
#         try:
#             # We pass ONLY the user_query and the target_source. No messy history.
#             response = system.run_pipeline(
#                 user_query=query,
#                 target_source=target_file
#             )

#             print(f"\n[Answer]: {response.answer}")
#             print(f"[Route]:  {response.lineage.route.upper()}")

#             if response.lineage.route == "sql" and response.lineage.sql_run:
#                 print(f"[SQL]:    {response.lineage.sql_run}")
#             if response.lineage.route == "rag" and response.lineage.documents_retrieved:
#                 print(f"[Docs]:   {response.lineage.documents_retrieved}")

#         except Exception as e:
#             print(f"Error: {e}")
#             import traceback
#             traceback.print_exc()

# if __name__ == "__main__":
#     # Keeping logs clean so the interactive terminal isn't spammed
#     import logging
#     logging.basicConfig(level=logging.WARNING)
#     run_demo()


"""
Main Pipeline - AI Query System
FIX: Replaced all hardcoded OpenAI model names with GROQ_MODELS
FIX: Added graceful fallback when Redis / DB is not available
FIX: Added detailed logging so you can see what each layer is doing
FIX: Cache now only stores results when sql_results is non-empty (prevents caching empty/broken runs)
"""

import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import os
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
        conn = None
        cursor = None
        try:
            # MUST HAVE THESE IMPORTS HERE!
            import psycopg2
            from pathlib import Path
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

            # Serialize setup work across concurrent app/process startups.
            cursor.execute("SELECT pg_advisory_lock(948302174221)")

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
                    sql = sql.replace("ALTER ROLE ai_readonly WITH PASSWORD '1234';", f"ALTER ROLE ai_readonly WITH PASSWORD '{db_pass}';")

                    cursor.execute(sql)
                    self.logger.info("Database auto-setup completed successfully!")
                else:
                    self.logger.warning("setup_db.sql not found. Cannot auto-setup database.")

        except Exception as e:
            # This will now tell us EXACTLY what broke if it fails again
            self.logger.error(f"Auto-setup completely failed: {e}")
        finally:
            if cursor is not None:
                try:
                    cursor.execute("SELECT pg_advisory_unlock(948302174221)")
                except Exception:
                    pass
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

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

    def upload_file(
        self,
        file_path: str,
        original_file_name: Optional[str] = None,
        user_email: Optional[str] = None,
        session_id: Optional[str] = None,
        upload_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload and process a single file (CSV/Excel/JSON -> SQL, PDF/TXT/DOCX -> RAG)."""
        if not self.doc_processor:
            return {"success": False, "message": "Document processor not initialized"}

        return self.doc_processor.process(
            file_path,
            original_file_name=original_file_name,
            user_email=user_email,
            session_id=session_id,
            upload_ts=upload_ts,
        )

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

    def _normalize_authorized_doc_names(self, authorized_docs: Optional[List[Any]]) -> List[str]:
        """Normalize mixed Mongo document records into plain file-name strings."""
        if not authorized_docs:
            return []

        normalized: List[str] = []
        for item in authorized_docs:
            if isinstance(item, dict):
                name = str(item.get("file_name", "")).strip()
            else:
                name = str(item).strip()

            if name and name not in normalized:
                normalized.append(name)

        return normalized

    def _build_doc_where_filter(
        self,
        user_email: Optional[str],
        authorized_docs: Optional[List[str]],
        target_source: Optional[str],
        context_filter: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        conditions = []
        normalized_docs = self._normalize_authorized_doc_names(authorized_docs)

        if user_email:
            conditions.append({"user_email": user_email})

        if authorized_docs is not None:
            if len(normalized_docs) == 0:
                conditions.append({"file_name": "__NO_AUTH__"})
            elif len(normalized_docs) == 1:
                conditions.append({"file_name": normalized_docs[0]})
            else:
                conditions.append({"file_name": {"$in": normalized_docs}})

        if target_source:
            conditions.append({"file_name": target_source})

        if context_filter:
            for key, val in context_filter.items():
                conditions.append({key: val})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _build_cache_key(
        self,
        user_query: str,
        route: str,
        user_email: Optional[str],
        target_source: Optional[str],
        context_filter: Optional[Dict[str, Any]],
        authorized_docs: Optional[List[str]],
    ) -> str:
        import hashlib
        import json

        normalized_docs = self._normalize_authorized_doc_names(authorized_docs)

        payload = { 
            "q": user_query,
            "route": route,
            "user_email": user_email or "",
            "target_source": target_source or "",
            "context_filter": context_filter or {},
            "authorized_docs": sorted(normalized_docs),
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def run_pipeline(
        self,
        user_query: str,
        context_filter: Optional[Dict[str, Any]] = None,
        authorized_docs: Optional[List[str]] = None,
        target_source: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> QueryResponse:
        start_time = time.time()
        self.logger.info(f"Query: {user_query}")

        # Step 1: Semantic Cache Check
        if self.cache:
            try:
                # route not finalized yet; use "pre_route" for lookup stage
                pre_route_cache_key = self._build_cache_key(
                    user_query=user_query,
                    route="pre_route",
                    user_email=user_email,
                    target_source=target_source,
                    context_filter=context_filter,
                    authorized_docs=authorized_docs,
                )
                if hasattr(self.cache, "get_exact"):
                    cached = self.cache.get_exact(pre_route_cache_key)
                else:
                    cached = self.cache.get(pre_route_cache_key)
                if cached and cached.get("answer"):
                    cached_meta = cached.get("metadata", {})
                    self.logger.info(f"[CACHE HIT] similarity={cached.get('similarity', 0):.3f}")
                    lineage = self.storyteller.create_lineage(
                        query=user_query,
                        route="cache",
                        cache_hit=True,
                        cache_similarity=cached.get("similarity"),
                        execution_time_ms=0,
                    )
                    return QueryResponse(
                        answer=cached["answer"],
                        lineage=lineage,
                        raw_results=cached_meta.get("results"),
                        raw_docs=cached_meta.get("docs"),
                        execution_error=cached_meta.get("execution_error"),
                    )
            except Exception as e:
                self.logger.warning(f"Cache lookup failed: {e}")

        # Step 2: Route query & Smart Override
        import re
        routing = self.router.route(user_query)
        route = routing.get("route", "sql")
        inferred_schemas = routing.get("schemas", [])

        if target_source:
            structured_exts = ['.csv', '.xlsx', '.xls', '.json']
            unstructured_exts = ['.pdf', '.txt', '.docx', '.md']

            target_lower = target_source.lower()
            if any(target_lower.endswith(ext) for ext in structured_exts):
                if route == "rag":
                    route = "sql"
                    self.logger.info(f"[ROUTER OVERRIDE] Forced SQL route for structured file: {target_source}")
            elif any(target_lower.endswith(ext) for ext in unstructured_exts):
                if route == "sql":
                    route = "rag"
                    self.logger.info(f"[ROUTER OVERRIDE] Forced RAG route for unstructured file: {target_source}")

        self.logger.info(f"[ROUTER] route={route} | inferred_schemas={inferred_schemas}")

        # Setup Search Term (Handling @mentions)
        mentions = re.findall(r"@([a-zA-Z0-9_.\-]+)", user_query)
        combined_hints = []
        for m in mentions:
            if m not in combined_hints: combined_hints.append(m)
        for sc in inferred_schemas:
            if sc not in combined_hints: combined_hints.append(sc)

        search_term = user_query
        if combined_hints:
            search_term = search_term + " " + " ".join(combined_hints)

        # Step 3: Retrieve schemas / documents
        schemas, docs, schema_context = [], [], ""

        # --- SQL SCHEMA RETRIEVAL (This was missing!) ---
        if route in ["sql", "both"]:
            schema_where = None
            if target_source:
                from pathlib import Path
                structured_exts = ['.csv', '.xlsx', '.xls', '.json']
                if any(target_source.lower().endswith(ext) for ext in structured_exts):
                    target_table = Path(target_source).stem.lower().replace(" ", "_").replace("-", "_")
                    schema_where = {"table_name": target_table}

            sql_top_k = int(self.config.get("tag", {}).get("top_k_schemas", 5))
            schemas = self.tag.retrieve_schemas(search_term, top_k=sql_top_k, where_filter=schema_where)
            schema_context = "\n\n".join([s.to_document()[:800] for s in schemas])
            self.logger.info(f"[TAG] Retrieved schemas: {[s.table_name for s in schemas]} with filter: {schema_where}")

        if route in ["rag", "both"]:
            where_filter = self._build_doc_where_filter(
                user_email=user_email,
                authorized_docs=authorized_docs,
                target_source=target_source,
                context_filter=context_filter,
            )

            docs = self.tag.retrieve_documents(search_term, top_k=12, where_filter=where_filter)
            docs = docs[:5]  # final context set
            self.logger.info(f"[TAG] Retrieved {len(docs)} documents with filter: {where_filter}")

        # Steps 4 & 5: Generate SQL and execute (Self-Healing Loop)
        sql_results, sql_query, tables_used = None, None, []
        execution_error = None

        if route in ["sql", "both"] and schema_context:
            if not self.executor:
                self.logger.warning("[SQL] Executor offline — skipping SQL execution. Check DB config.")
            else:
                self.logger.info("[SQL ENGINE] Generating SQL via multi-agent pipeline...")

                max_retries = 3
                feedback_history = ""

                for attempt in range(max_retries):
                    self.logger.info(f"[SQL ENGINE] Attempt {attempt + 1}/{max_retries}...")

                    enhanced_query = user_query
                    if feedback_history:
                        enhanced_query += f"\n\nDO NOT repeat previous mistakes. Previous attempts failed with:\n{feedback_history}"

                    sql_result = self.sql_engine.execute(enhanced_query, schema_context)

                    if sql_result.success:
                        sql_query = sql_result.query
                        tables_used = sql_result.tables_used
                        self.logger.info(f"[SQL ENGINE] Generated SQL:\n{sql_query}")

                        try:
                            db_result = self.executor.execute(sql_query)
                            if db_result.success:
                                sql_results = db_result.rows
                                execution_error = None
                                self.logger.info(f"[EXECUTOR] Got {len(sql_results)} rows in {db_result.execution_time_ms:.1f}ms")
                                break # Success!
                            else:
                                execution_error = db_result.error
                                self.logger.warning(f"[EXECUTOR] DB error: {db_result.error}. Retrying...")
                                feedback_history += f"- Query: {sql_query}\n- Error: {db_result.error}\n"
                        except Exception as e:
                            execution_error = str(e)
                            self.logger.error(f"[EXECUTOR] Exception: {e}")
                            break
                    else:
                        self.logger.warning(f"[SQL ENGINE] Validation failed: {sql_result.validation_errors}")
                        feedback_history += f"- Validation Error: {sql_result.validation_errors}\n"

                if not sql_results and feedback_history:
                    self.logger.error("[SQL ENGINE] All retry attempts failed.")

        # --- 🕵️ HACKATHON DEBUG: WHAT IS THE AI ACTUALLY SEEING? ---
        if route in ["rag", "both"]:
            print("\n" + "!" * 50)
            print("🕵️ X-RAY VISION: RAG DOCUMENTS RETRIEVED")
            print("!" * 50)
            if not docs:
                print("❌ ERROR: 0 chunks retrieved! The ChromaDB filter failed or the PDF is empty.")
            else:
                for i, d in enumerate(docs):
                    print(f"\n--- CHUNK {i+1} (Score: {d.get('distance', 'N/A')}) ---")
                    # Print the first 400 characters of what pypdf actually extracted
                    print(d.get('content', '')[:400])
            print("!" * 50 + "\n")
        # -----------------------------------------------------------

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

        if self.cache and (sql_results or docs):
            try:
                route_cache_key = self._build_cache_key(
                    user_query=user_query,
                    route=route,
                    user_email=user_email,
                    target_source=target_source,
                    context_filter=context_filter,
                    authorized_docs=authorized_docs,
                )
                pre_route_cache_key = self._build_cache_key(
                    user_query=user_query,
                    route="pre_route",
                    user_email=user_email,
                    target_source=target_source,
                    context_filter=context_filter,
                    authorized_docs=authorized_docs,
                )
                cache_metadata = {
                    "route": route,
                    "results": sql_results,
                    "docs": docs,
                    "execution_error": execution_error,
                    "docs_count": len(docs),
                }
                if hasattr(self.cache, "set_exact"):
                    self.cache.set_exact(
                        route_cache_key,
                        answer,
                        metadata=cache_metadata,
                    )
                    self.cache.set_exact(pre_route_cache_key, answer, metadata=cache_metadata)
                else:
                    self.cache.set(
                        route_cache_key,
                        answer,
                        metadata=cache_metadata,
                    )
                    self.cache.set(pre_route_cache_key, answer, metadata=cache_metadata)
                self.logger.info("[CACHE] Stored isolated cache entry.")

            except Exception as e:
                self.logger.warning(f"Cache write failed: {e}")

        self.storyteller.log_lineage(lineage)
        return QueryResponse(
            answer=answer,
            lineage=lineage,
            raw_results=sql_results,
            raw_docs=docs,
            execution_error=execution_error,
        )

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
                if not meta:
                    continue
                if "file_name" in meta and meta["file_name"]:
                    sources.add(meta["file_name"])
                elif "source" in meta and meta["source"]:
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
    import os

    print("=" * 60)
    print("AI Query System - STRICT TARGETING DEMO")
    print("=" * 60)

    system = AIQuerySystem(load_sample_schemas=True)
    cleared = system.clear_cache()
    if cleared:
        print(f"\nCleared {cleared} stale cache entries")

    # ==========================================
    # STEP 1: UPLOAD FILES
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 1: Upload Your Documents")
    print("=" * 60)

    active_files = []

    unstructured_file = input("Path to an unstructured file (e.g., ./docs/manual.pdf) [Enter to skip]: ").strip()
    if unstructured_file and os.path.exists(unstructured_file):
        res = system.upload_file(unstructured_file)
        if res.get('success'):
            active_files.append(os.path.basename(unstructured_file))
            print(f" -> Success! Added {os.path.basename(unstructured_file)}")
        else:
            print(f" -> Failed: {res.get('message')}")

    structured_file = input("Path to a structured file (e.g., ./data/inventory.csv) [Enter to skip]: ").strip()
    if structured_file and os.path.exists(structured_file):
        res = system.upload_file(structured_file)
        if res.get('success'):
            active_files.append(os.path.basename(structured_file))
            print(f" -> Success! Added {os.path.basename(structured_file)}")
        else:
            print(f" -> Failed: {res.get('message')}")

    if not active_files:
        print("\nNo files were successfully uploaded. Exiting demo.")
        return

    # ==========================================
    # STEP 2: FORCED INTERACTIVE LOOP
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 2: Strict Querying")
    print("=" * 60)

    while True:
        print("\nAVAILABLE FILES:")
        for i, f in enumerate(active_files, 1):
            print(f"[{i}] {f}")
        print(f"[{len(active_files) + 1}] Exit Demo")

        # 1. FORCE THE FILE SELECTION FIRST
        file_choice = input("\nWhich file do you want to query? (Enter the number): ").strip()

        if file_choice == str(len(active_files) + 1):
            print("Exiting demo...")
            break

        try:
            choice_idx = int(file_choice) - 1
            if choice_idx < 0 or choice_idx >= len(active_files):
                print("Invalid number. You MUST select a valid file from the list.")
                continue
            target_file = active_files[choice_idx]
        except ValueError:
            print("Please enter a valid number.")
            continue

        # 2. ONLY ASK FOR QUERY AFTER FILE IS SECURED
        query = input(f"\nEnter your query specifically for '{target_file}': ").strip()
        if not query:
            continue

        # 3. EXECUTE WITH GUARANTEED CONTEXT
        print(f"\n--- [Locking pipeline context strictly to: {target_file}] ---")
        try:
            # We pass ONLY the user_query and the target_source. No messy history.
            response = system.run_pipeline(
                user_query=query,
                target_source=target_file
            )

            print(f"\n[Answer]: {response.answer}")
            print(f"[Route]:  {response.lineage.route.upper()}")

            if response.lineage.route == "sql" and response.lineage.sql_run:
                print(f"[SQL]:    {response.lineage.sql_run}")
            if response.lineage.route == "rag" and response.lineage.documents_retrieved:
                print(f"[Docs]:   {response.lineage.documents_retrieved}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Keeping logs clean so the interactive terminal isn't spammed
    import logging
    logging.basicConfig(level=logging.WARNING)
    run_demo()