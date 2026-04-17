"""
Document Processor
Handles uploaded files and routes them to the correct pipeline:
  - Structured data (CSV, Excel, JSON) → PostgreSQL + TAG schema
  - Unstructured docs (PDF, TXT, DOCX, MD) → ChromaDB RAG collection
"""

import os
import json
import uuid
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)



STRUCTURED_EXTENSIONS   = {".csv", ".xlsx", ".xls", ".json"}
UNSTRUCTURED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md", ".markdown"}


def classify_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in STRUCTURED_EXTENSIONS:
        return "structured"
    if ext in UNSTRUCTURED_EXTENSIONS:
        return "unstructured"
    return "unsupported"


# ---------------------------------------------------------------------------
# Structured data handlers
# ---------------------------------------------------------------------------

class StructuredFileLoader:
    def load(self, file_path: str, original_file_name: Optional[str] = None):
        import pandas as pd

        file_name = original_file_name or Path(file_path).name
        ext  = Path(file_name).suffix.lower()
        stem = Path(file_name).stem.lower().replace(" ", "_").replace("-", "_")

        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path)
        elif ext == ".json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported structured format: {ext}")

        # Sanitize column names
        df.columns = [
            c.lower().strip().replace(" ", "_").replace("-", "_")
            for c in df.columns
        ]

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {file_name}")
        return df, stem

    def infer_postgres_type(self, dtype) -> str:
        dtype_str = str(dtype)
        if "int" in dtype_str:
            return "BIGINT"
        if "float" in dtype_str:
            return "DOUBLE PRECISION"
        if "bool" in dtype_str:
            return "BOOLEAN"
        if "datetime" in dtype_str:
            return "TIMESTAMP"
        if "date" in dtype_str:
            return "DATE"
        return "TEXT"

    def create_table_and_insert(
        self,
        df,
        table_name: str,
        engine,
        if_exists: str = "replace"
    ) -> bool:
        from sqlalchemy import text
        try:
            df.to_sql(table_name, engine, if_exists=if_exists, index=False, method="multi", chunksize=500)

            with engine.connect() as conn:
                quoted_table = engine.dialect.identifier_preparer.quote(table_name)

                conn.execute(text(f"GRANT SELECT ON TABLE {quoted_table} TO ai_readonly;"))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to write table '{table_name}': {e}")
            return False

    def build_table_description(self, df, table_name: str, file_name: str):
        from layers.layer3_tag import TableDescription
        columns = []
        sample_values = {}

        for col in df.columns:
            pg_type = self.infer_postgres_type(df[col].dtype)
            columns.append({
                "name":        col,
                "type":        pg_type,
                "description": f"Column from uploaded file {file_name}"
            })
            non_null = df[col].dropna()
            if len(non_null) > 0:
                sample_values[col] = str(non_null.iloc[0])

        return TableDescription(
            table_name=table_name,
            description=(
                f"Uploaded structured dataset from '{file_name}'. "
                f"Contains {len(df)} rows and {len(df.columns)} columns."
            ),
            columns=columns,
            relationships=[],
            sample_values=sample_values
        )


# ---------------------------------------------------------------------------
# Unstructured document handlers
# ---------------------------------------------------------------------------

class UnstructuredFileLoader:
    def load(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext == ".txt" or ext in {".md", ".markdown"}:
            return self._load_text(file_path)
        elif ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext == ".docx":
            return self._load_docx(file_path)
        else:
            raise ValueError(f"Unsupported unstructured format: {ext}")

    def _load_text(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _load_pdf(self, file_path: str) -> str:
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # x_tolerance and y_tolerance force the library to read scattered math text
                    text = page.extract_text(x_tolerance=2, y_tolerance=3)
                    if text:
                        pages.append(text)

            full_text = "\n\n".join(pages)

            # Hackathon Safety Check
            if len(full_text.strip()) < 200:
                print("\nWARNING: Almost no text extracted! If this is a scanned PDF (an image), you will need OCR (pytesseract) to read it.\n")

            return full_text

        except ImportError:
            raise ImportError("Please run: pip install pdfplumber")
        except Exception as e:
            logger.error(f"Failed to read PDF {file_path}: {e}")
            return ""

    def _load_docx(self, file_path: str) -> str:
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except ImportError:
            raise ImportError("python-docx is required for DOCX support.")

    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        words  = text.split()
        chunks = []
        start  = 0

        while start < len(words):
            end   = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            if end == len(words):
                break
            start = end - chunk_overlap
        return chunks


# ---------------------------------------------------------------------------
# Main DocumentProcessor
# ---------------------------------------------------------------------------

class DocumentProcessor:
    def __init__(
        self,
        tag,
        executor=None,
        admin_db_url: str = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.tag          = tag
        self.executor     = executor
        self.chunk_size   = chunk_size
        self.chunk_overlap = chunk_overlap

        self._structured_loader   = StructuredFileLoader()
        self._unstructured_loader = UnstructuredFileLoader()

        self._admin_engine = None
        if admin_db_url:
            from sqlalchemy import create_engine
            self._admin_engine = create_engine(admin_db_url, pool_pre_ping=True)

    def process(self, file_path: str, original_file_name: Optional[str] = None) -> Dict[str, Any]:
        """Fix: properly handles original_file_name mapping"""
        file_path = str(file_path)
        file_name = original_file_name or Path(file_path).name
        file_type = classify_file(file_name)

        if file_type == "structured":
            return self._process_structured(file_path, file_name)
        elif file_type == "unstructured":
            return self._process_unstructured(file_path, file_name)
        else:
            return {
                "success":   False,
                "file_type": "unsupported",
                "file_name": file_name,
                "message":   f"Unsupported file type: {Path(file_name).suffix}."
            }

    def process_many(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        return [self.process(p) for p in file_paths]

    def list_loaded_schemas(self) -> List[str]:
        try:
            count = self.tag.schema_collection.count()
            if count == 0:
                return []
            results = self.tag.schema_collection.get()
            return results.get("ids", [])
        except Exception:
            return []

    def list_loaded_documents(self) -> List[Dict[str, str]]:
        try:
            count = self.tag.docs_collection.count()
            if count == 0:
                return []
            results = self.tag.docs_collection.get()
            ids       = results.get("ids", [])
            metadatas = results.get("metadatas", [{}] * len(ids))
            return [
                {"id": id_, "file_name": m.get("file_name", "unknown")}
                for id_, m in zip(ids, metadatas)
            ]
        except Exception:
            return []

    def _process_structured(self, file_path: str, file_name: str) -> Dict[str, Any]:
        try:
            df, table_name = self._structured_loader.load(file_path, file_name)

            db_written = False
            if self._admin_engine:
                db_written = self._structured_loader.create_table_and_insert(
                    df, table_name, self._admin_engine
                )

            table_desc = self._structured_loader.build_table_description(
                df, table_name, file_name
            )
            self.tag.add_schema(table_desc)

            return {
                "success":    True,
                "file_type":  "structured",
                "file_name":  file_name,
                "table_name": table_name,
                "row_count":  len(df),
                "columns":    list(df.columns),
                "db_written": db_written,
                "message":    f"Loaded '{table_name}' and added schema to TAG."
            }
        except Exception as e:
            return {
                "success":   False,
                "file_type": "structured",
                "file_name": file_name,
                "message":   f"Failed: {str(e)}"
            }

    def _process_unstructured(self, file_path: str, file_name: str) -> Dict[str, Any]:
        try:
            text = self._unstructured_loader.load(file_path)
            if not text.strip():
                return {"success": False, "file_type": "unstructured", "file_name": file_name, "message": "Empty file"}

            chunks = self._unstructured_loader.chunk_text(
                text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )

            file_hash = hashlib.md5(file_name.encode()).hexdigest()[:8]
            doc_ids   = []

            for i, chunk in enumerate(chunks):
                doc_id = f"{file_hash}_chunk_{i}"
                self.tag.add_document(
                    doc_id=doc_id,
                    content=chunk,
                    metadata={
                        "file_name":  file_name,
                        "chunk_index": str(i),
                        "total_chunks": str(len(chunks)),
                        "file_type":  Path(file_name).suffix.lower()
                    }
                )
                doc_ids.append(doc_id)

            return {
                "success":     True,
                "file_type":   "unstructured",
                "file_name":   file_name,
                "chunk_count": len(chunks),
                "doc_ids":     doc_ids,
                "char_count":  len(text),
                "message":     f"Extracted chunks from '{file_name}'"
            }
        except Exception as e:
            return {
                "success":   False,
                "file_type": "unstructured",
                "file_name": file_name,
                "message":   f"Failed: {str(e)}"
            }


def create_document_processor(tag, executor=None, config: Dict[str, Any] = None) -> DocumentProcessor:
    config = config or {}
    db_host   = config.get("db_host",  os.getenv("DB_HOST",  "localhost"))
    db_port   = config.get("db_port",  int(os.getenv("DB_PORT", "5432")))
    db_name   = config.get("db_name",  os.getenv("DB_NAME",  "postgres"))

    admin_user = (config.get("admin_db_user") or os.getenv("ADMIN_DB_USER") or os.getenv("DB_USER", "postgres"))
    admin_pass = (config.get("admin_db_password") or os.getenv("ADMIN_DB_PASSWORD") or os.getenv("DB_PASSWORD", ""))

    admin_db_url = None
    if admin_user and admin_pass:
        admin_db_url = f"postgresql://{admin_user}:{admin_pass}@{db_host}:{db_port}/{db_name}"

    return DocumentProcessor(
        tag=tag,
        executor=executor,
        admin_db_url=admin_db_url,
        chunk_size=config.get("chunk_size", 500),
        chunk_overlap=config.get("chunk_overlap", 50)
    )
