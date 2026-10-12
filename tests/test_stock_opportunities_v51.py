"""Stock tags select bounded inspection opportunities, never approve a shot."""
from copy import deepcopy

import pytest

from jinggao import pooled_match, stock_queries


@pytest.mark.parametrize(("query", "tags", "expected"), [
    ("chopping vegetables on cutting board", "chopping, cutting, ingredients, garlic, food, kitchen", {"chopping", "cutting"}),
    ("washing vegetables kitchen home", "wash, vegetable, sink", {"washing", "vegetables"}),
    ("stir frying vegetables wok", "stir, fry, food, cooking", {"stir", "frying"}),
    ("people walking in park", "people, walk, park", {"people", "walking", "park"}),
    ("washing vegetables kitchen home", "pot, cooking, kitchen, steam", set()),
    ("on the close up", "on the close up", set()),
    ("chopping vegetables", "vegetables on shop shelf", set()),
])
def test_partial_tag_opportunities_keep_strict_full_match_separate(query, tags, expected):
    assert set(stock_queries.metadata_overlap(query, tags)) == expected
    assert stock_queries.metadata_matches(query, tags) == (query == "on the close up")


def test_stock_slots_remain_bounded_and_do_not_change_candidates_or_conditions(monkeypatch):
    monkeypatch.setattr(pooled_match, "stock_search_query", lambda *args: "chopping vegetables on cutting board")
    group = {"indices": [0, 1], "queries": ["案板切菜"]}
    domestic = [{"id": f"domestic-{i}", "provider": "bilibili", "title": "切菜"} for i in range(12)]
    stocks = [{"id": f"stock-{i}", "provider": "pixabay", "title": "chopping, cutting, garlic",
               "status": "unverified", "approved": False} for i in range(20)]
    rows = [*domestic, *stocks]
    prior = deepcopy(rows)
    before = deepcopy(group)
    selected, notes = pooled_match.stock_candidate_opportunities({}, group, [], rows, domestic,
        [{"candidate_id": row["id"]} for row in domestic], 12, [0, 1])
    assert len(selected) == 12 and sum(c["provider"] == "pixabay" for c in selected) == 4
    assert rows == prior and group == before
    assert all(not c.get("approved") for c in selected)
    assert all(n["metadata_only"] and n["stock_queries"][0]["matched_tag_terms"] == ["chopping", "cutting"]
               for n in notes if n.get("stock_queries"))


def test_named_or_stale_stock_hint_cannot_open_stock_slots(monkeypatch):
    monkeypatch.setattr(pooled_match, "stock_search_query", lambda *args: None)
    chosen = [{"id": "named-person"}]
    audit = [{"candidate_id": "named-person"}]
    stock = [{"id": "wrong", "provider": "pixabay", "title": "chopping cutting"}]
    assert pooled_match.stock_candidate_opportunities({}, {"queries": ["具名人物"]}, [], stock,
        chosen, audit, 3, [0]) == (chosen, audit)
