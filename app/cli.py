"""Command line interface for local handbook ingestion and question answering."""
import argparse
import logging
from pathlib import Path
import re
import sys

from app.config import settings
from app.embeddings.embedding_manager import EmbeddingManager
from app.generation.ollama_client import OllamaClient
from app.generation.prompt_builder import ABSTENTION_MESSAGE, build_rag_messages
from app.ingestion.chunker import DocumentChunker
from app.ingestion.metadata_extractor import extract_programme_info
from app.ingestion.pdf_loader import PDFLoader
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.keyword_retriever import KeywordRetriever
from app.storage.vector_store import VectorStore
from rich.console import Console
from rich.markdown import Markdown

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger=logging.getLogger(__name__)
console = Console()

def _format_markdown(text):
    """Normalize common model list formatting before rendering Markdown."""
    lines=[]
    for line in text.strip().splitlines():
        match=re.match(r"^(\s*)\+\s+(.*)$",line)
        if match:
            # Models often indent nested bullets by eight spaces, which Markdown
            # treats as a code block. Render them at the conventional list indent.
            line=f"  - {match.group(2)}"
        lines.append(line)
    return "\n".join(lines)

def _print_answer(answer,hits):
    console.rule("Answer",style="cyan")
    console.print(Markdown(_format_markdown(answer)))
    sources=sorted({(h.get("source_filename",h.get("source","unknown")),h.get("page_number","?")) for h in hits})
    console.print("\n[bold]Retrieved sources[/bold]")
    for filename,page in sources:
        console.print(f"  • {filename}, page {page}")
    console.print()

def _components(db_dir=None):
    embed=EmbeddingManager(settings.embedding_model)
    store=VectorStore(db_dir or settings.db_dir, embedding_model_name=settings.embedding_model, embedding_dimension=embed.dimension or 384)
    return embed,store

def ingest(paths,force_reindex=False,db_dir=None):
    embed,store=_components(db_dir); loader=PDFLoader(); chunker=DocumentChunker(settings.chunk_size,settings.chunk_overlap)
    total=0
    for path in paths:
        doc=loader.load_pdf(path); filename=doc.path.name
        if force_reindex: store.delete_by_source(filename)
        existing=set(store.get_sources())
        if filename in existing and not force_reindex:
            logger.info("Already indexed; skipping %s (use --force-reindex to replace)",filename); continue
        programme,language=extract_programme_info(filename,doc.pages[0].text if doc.pages else "")
        chunks=[]
        for page in doc.pages:
            chunks.extend(chunker.chunk_page(page.text,filename,programme,page.page_number,doc.file_hash,language))
        if not chunks: logger.warning("No text chunks found in %s",filename); continue
        vectors=embed.encode_passages([c.text for c in chunks]); total+=store.add_chunks(chunks,vectors)
        logger.info("Indexed %s: %d pages, %d chunks",filename,doc.total_pages,len(chunks))
    print(f"Indexed {total} chunks")

def ask(question,top_k=6,db_dir=None):
    embed,store=_components(db_dir)
    docs=store.get_all_chunks(); retriever=HybridRetriever(DenseRetriever(store,embed),KeywordRetriever(docs))
    hits=retriever.retrieve(question,top_k=top_k)
    if not hits:
        print(ABSTENTION_MESSAGE); return
    client=OllamaClient(settings.ollama_url)
    model=client.select_best_model([settings.ollama_model]) or settings.ollama_model
    answer=client.generate_chat(build_rag_messages(question,hits),model=model)
    _print_answer(answer,hits)

def chat(top_k=6,db_dir=None):
    print("Saarland University module handbook assistant. Type 'quit' to exit.")
    while True:
        try:
            question=input("You: ").strip()
        except (EOFError,KeyboardInterrupt):
            print("\nGoodbye."); return
        if question.lower() in {"quit","exit","q"}: return
        if question:
            try: ask(question,top_k,db_dir)
            except (ConnectionError,TimeoutError) as exc: print(f"error: {exc}")

def main(argv=None):
    parser=argparse.ArgumentParser(description="Saarland University module handbook assistant")
    parser.add_argument("--top-k",type=int,default=settings.default_top_k)
    parser.add_argument("--db-dir",type=Path,default=settings.db_dir)
    sub=parser.add_subparsers(dest="command",required=True)
    pi=sub.add_parser("ingest",help="Index PDF handbooks"); pi.add_argument("pdfs",nargs="+"); pi.add_argument("--force-reindex",action="store_true")
    pa=sub.add_parser("ask",help="Ask one question"); pa.add_argument("question",nargs="+")
    sub.add_parser("chat",help="Start an interactive question session")
    sub.add_parser("status",help="Show indexed database status")
    sub.add_parser("setup",help="Show setup commands")
    args=parser.parse_args(argv)
    try:
        if args.command=="ingest":
            ingest(args.pdfs,args.force_reindex,args.db_dir)
        elif args.command=="ask":
            ask(" ".join(args.question),args.top_k,args.db_dir)
        elif args.command=="chat":
            chat(args.top_k,args.db_dir)
        elif args.command=="status":
            _,store=_components(args.db_dir); print(f"Database: {args.db_dir}\nChunks: {store.count()}\nSources: {', '.join(store.get_sources()) or '(none)'}")
        elif args.command=="setup":
            print("Install requirements, run 'ollama serve', pull a model, then run: python rag.py ingest data/visual_computing.pdf")
    except (ValueError,FileNotFoundError,ConnectionError,TimeoutError) as exc:
        parser.exit(1,f"error: {exc}\n")
