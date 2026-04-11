"""
Layer 3: Table Augmented Generation (TAG)
FIX: ChromaDB metadata cannot contain nested dicts/lists of dicts.
     Columns are now serialized to a JSON string before storage,
     and deserialized on retrieval.
"""

import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os


class TableDescription:
    def __init__(
        self,
        table_name: str,
        description: str,
        columns: List[Dict[str, str]],
        relationships: Optional[List[str]] = None,
        sample_values: Optional[Dict[str, Any]] = None
    ):
        self.table_name = table_name
        self.description = description
        self.columns = columns
        self.relationships = relationships or []
        self.sample_values = sample_values or {}

    def to_document(self) -> str:
        """Convert table description to a searchable document string."""
        doc = f"Table: {self.table_name}\n"
        doc += f"Description: {self.description}\n"
        doc += "Columns:\n"
        for col in self.columns:
            doc += f"  - {col['name']}: {col['type']}"
            if col.get('description'):
                doc += f" ({col['description']})"
            if col['name'] in self.sample_values:
                doc += f" [e.g., {self.sample_values[col['name']]}]"
            doc += "\n"
        if self.relationships:
            doc += "Relationships:\n"
            for rel in self.relationships:
                doc += f"  - {rel}\n"
        return doc

    def to_metadata(self) -> Dict[str, Any]:
        """
        FIX: ChromaDB only accepts str/int/float/bool in metadata values.
        Serialize lists and dicts to JSON strings.
        """
        return {
            "table_name": self.table_name,
            "description": self.description,
            "columns_json": json.dumps(self.columns),              # list of dicts → JSON string
            "relationships_json": json.dumps(self.relationships),  # list → JSON string
            "sample_values_json": json.dumps(self.sample_values),  # dict → JSON string
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "description": self.description,
            "columns": self.columns,
            "relationships": self.relationships,
            "sample_values": self.sample_values
        }

    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any]) -> "TableDescription":
        """FIX: Deserialize JSON strings back to Python objects on retrieval."""
        return cls(
            table_name=metadata["table_name"],
            description=metadata["description"],
            columns=json.loads(metadata.get("columns_json", "[]")),
            relationships=json.loads(metadata.get("relationships_json", "[]")),
            sample_values=json.loads(metadata.get("sample_values_json", "{}"))
        )


class TAGRetrieval:
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.schema_collection = self.client.get_or_create_collection(
            name="schema_metadata",
            metadata={"description": "Database schema metadata"}
        )
        self.docs_collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "Document RAG collection"}
        )

    def add_schema(self, table: TableDescription) -> str:
        doc = table.to_document()
        embedding = self.model.encode(doc).tolist()

        # FIX: Use to_metadata() which serializes nested structures
        self.schema_collection.upsert(   # upsert avoids duplicate-key errors on re-runs
            documents=[doc],
            embeddings=[embedding],
            ids=[table.table_name],
            metadatas=[table.to_metadata()]
        )
        return table.table_name

    def add_schemas(self, tables: List[TableDescription]) -> List[str]:
        return [self.add_schema(table) for table in tables]

    def retrieve_schemas(self, query: str, top_k: int = 3) -> List[TableDescription]:
        query_embedding = self.model.encode(query).tolist()
        results = self.schema_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.schema_collection.count() or 1)
        )

        tables = []
        if results and results["ids"]:
            for i, _ in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                # FIX: Use from_metadata() to deserialize JSON strings back
                tables.append(TableDescription.from_metadata(metadata))
        return tables

    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        embedding = self.model.encode(content).tolist()
        # FIX: Ensure metadata values are only primitives
        safe_metadata = {k: str(v) for k, v in (metadata or {}).items()}
        self.docs_collection.upsert(
            documents=[content],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[safe_metadata]
        )
        return doc_id

    def retrieve_documents(self, query: str, top_k: int = 5, where_filter: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        if self.docs_collection.count() == 0:
            return []
        query_embedding = self.model.encode(query).tolist()
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.docs_collection.count())
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.docs_collection.query(**query_kwargs)
        docs = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                docs.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
        return docs

    def clear_schema_collection(self) -> bool:
        try:
            self.client.delete_collection("schema_metadata")
            self.schema_collection = self.client.get_or_create_collection("schema_metadata")
            return True
        except Exception:
            return False

    def clear_docs_collection(self) -> bool:
        try:
            self.client.delete_collection("documents")
            self.docs_collection = self.client.get_or_create_collection("documents")
            return True
        except Exception:
            return False


def create_sample_schemas() -> List[TableDescription]:
    return [
        TableDescription(
            table_name="customers",
            description="Customer information including contact details and demographics",
            columns=[
                {"name": "customer_id", "type": "INTEGER PRIMARY KEY", "description": "Unique customer identifier"},
                {"name": "name",        "type": "VARCHAR(255)",        "description": "Customer full name"},
                {"name": "email",       "type": "VARCHAR(255)",        "description": "Customer email address"},
                {"name": "region",      "type": "VARCHAR(100)",        "description": "Geographic region"},
                {"name": "created_at",  "type": "TIMESTAMP",           "description": "Account creation date"}
            ],
            relationships=["customers.customer_id -> orders.customer_id (one-to-many)"],
            sample_values={"region": "North America", "customer_id": 12345}
        ),
        TableDescription(
            table_name="orders",
            description="Order transactions including order details and status",
            columns=[
                {"name": "order_id",      "type": "INTEGER PRIMARY KEY", "description": "Unique order identifier"},
                {"name": "customer_id",   "type": "INTEGER",             "description": "Foreign key to customers"},
                {"name": "order_date",    "type": "DATE",                "description": "Date order was placed"},
                {"name": "total_amount",  "type": "DECIMAL(10,2)",       "description": "Total order amount"},
                {"name": "status",        "type": "VARCHAR(50)",         "description": "Order status: pending, completed, cancelled"}
            ],
            relationships=["orders.customer_id -> customers.customer_id (many-to-one)"],
            sample_values={"total_amount": 299.99, "status": "completed"}
        ),
        TableDescription(
            table_name="products",
            description="Product catalog with pricing and inventory",
            columns=[
                {"name": "product_id",       "type": "INTEGER PRIMARY KEY", "description": "Unique product identifier"},
                {"name": "name",             "type": "VARCHAR(255)",        "description": "Product name"},
                {"name": "category",         "type": "VARCHAR(100)",        "description": "Product category"},
                {"name": "price",            "type": "DECIMAL(10,2)",       "description": "Unit price"},
                {"name": "inventory_count",  "type": "INTEGER",             "description": "Current stock level"}
            ],
            relationships=[],
            sample_values={"price": 49.99, "category": "Electronics"}
        )
    ]


def create_tag_retrieval(config: Dict[str, Any]) -> TAGRetrieval:
    return TAGRetrieval(
        persist_directory=config.get("chroma_persist_dir", "./data/chroma_db"),
        embedding_model=config.get("semantic_cache", {}).get("embedding_model", "all-MiniLM-L6-v2")
    )


if __name__ == "__main__":
    tag = TAGRetrieval()
    schemas = create_sample_schemas()
    tag.add_schemas(schemas)
    print(f"Added {len(schemas)} schemas")

    for query in ["Show me customer order information", "What products do we have?"]:
        results = tag.retrieve_schemas(query, top_k=2)
        print(f"Query: {query} → {[t.table_name for t in results]}")
