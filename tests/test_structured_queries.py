"""Parameterized structured query tests for module records."""
from app.storage.structured_store import StructuredModuleStore


def test_structured_filters_and_parameterized_name_search(tmp_path):
    store=StructuredModuleStore(tmp_path / "modules.sqlite3")
    assert store.count_modules() == 0
    # Search strings are values, not SQL fragments.
    assert store.find_module_by_code_or_name("' OR 1=1 --") == []
    assert store.list_modules(ects=9,limit=5) == []
