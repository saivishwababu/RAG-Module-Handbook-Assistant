"""Small deterministic retrieval evaluation runner for labelled question sets."""
import argparse
import json
from pathlib import Path
from typing import Any


def evaluate(retriever: Any, questions: list[dict], top_k: int = 6) -> dict:
    evaluated=[]; hits=0
    for item in questions:
        results=retriever.retrieve(item["question"],top_k=top_k)
        expected=set(item.get("expected_sources",[]))
        found={h.get("source_filename",h.get("source")) for h in results}
        hit=bool(expected & found) if expected else bool(results)
        hits+=int(hit)
        evaluated.append({"question":item["question"],"hit":hit,"expected_sources":sorted(expected),"retrieved_sources":sorted(x for x in found if x)})
    return {"count":len(questions),"top_k":top_k,"source_hit_rate":hits/len(questions) if questions else 0.0,"cases":evaluated}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions",nargs="?",type=Path,default=Path(__file__).with_name("questions.json"))
    parser.add_argument("--top-k",type=int,default=6)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    from app.cli import _components
    from app.retrieval.dense_retriever import DenseRetriever
    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.keyword_retriever import KeywordRetriever
    embed,store=_components()
    retriever=HybridRetriever(DenseRetriever(store,embed),KeywordRetriever(store.get_all_chunks()))
    report=evaluate(retriever,json.loads(args.questions.read_text()),args.top_k)
    payload=json.dumps(report,indent=2,ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(payload+"\n")
    print(payload)

if __name__=="__main__": main()
