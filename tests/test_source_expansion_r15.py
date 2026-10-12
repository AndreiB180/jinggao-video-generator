"""Replay saved R15 coordinates/queues without model, network or project edits."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

from jinggao import pooled_match, source_expansion


def evidence():
    return json.loads((Path(__file__).parent / "fixtures/r15_source_expansion.json").read_text())


def test_r15_edge_seed_repair_moves_beyond_all_four_initial_positions():
    case = evidence()["edge_seed"]
    original = deepcopy(case)
    prior = [(w["start"], w["end"]) for w in case["initial_probes"]]
    assert len(prior) == 4 and len(case["actual_repair_probes"]) == 1
    windows = source_expansion.neighbour_windows(case["candidate"], case["verified"], case["occupied"],
                                                 case["durations"], probed=prior)
    assert len(windows) == 4 and case == original
    assert all(w["start"] > max(end for _, end in prior) for w in windows)
    assert all(not (w["start"] < end and w["end"] > start)
               for w in windows for start, end in [*case["occupied"], *prior])
    ranges = sorted((w["start"], w["end"]) for w in windows)
    assert all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:], strict=False))
    assert all(w["end"] < case["candidate"]["duration"] for w in windows)


def test_r15_repair_prioritizes_new_reliable_source_before_old_unused_context():
    case = evidence()["repair_queue"]
    original = deepcopy(case)
    attempted = {pooled_match.proposal_location(case["failed_window"], case["candidates"])}
    ordered = pooled_match.prioritize_observed_proposals(case["previous"], case["current"], case["candidates"],
                                                        set(case["reliable_candidate_ids"]), attempted)
    # The real R15 run instead spent its last attempt on previous[1], whose
    # final review rejected unrelated film-award / romance captions.
    assert ordered[0] == case["current"][0] and ordered[1] == case["current"][1]
    assert ordered[2] == case["previous"][1] and case == original
    candidate = case["candidates"][ordered[0]["candidate_id"]]
    used = defaultdict(list, deepcopy(case["selected_source_intervals"]))
    slot = pooled_match.reserve_sampled(candidate, ordered[0], case["duration"], used)
    assert slot is not None and slot[0] == 56.42
    # Only a location is allocated. The protocol does not manufacture review.
    assert ordered[0]["sampling_only"] is True
    assert not {"approved", "verification", "verdict"} & ordered[0].keys()


def test_failed_window_cannot_return_as_new_reliable_candidate_alias():
    case = evidence()["repair_queue"]
    failed = case["previous"][0]
    alias = failed | {"candidate_id": "alias-for-the-same-public-window"}
    candidates = case["candidates"] | {alias["candidate_id"]: case["candidates"][failed["candidate_id"]]}
    attempted = {pooled_match.proposal_location(failed, candidates)}
    ordered = pooled_match.prioritize_observed_proposals(
        case["previous"], [alias, *case["current"], *case["current"]], candidates,
        {*case["reliable_candidate_ids"], alias["candidate_id"]}, attempted)
    locations = [pooled_match.proposal_location(p, candidates) for p in ordered]
    assert len(locations) == len(set(locations)) and not set(locations) & attempted
    assert alias not in ordered and failed not in ordered
    assert ordered[0] == case["current"][0]


def test_unqualified_current_source_does_not_jump_the_existing_locator_order():
    case = evidence()["repair_queue"]
    attempted = {pooled_match.proposal_location(case["failed_window"], case["candidates"])}
    ordered = pooled_match.prioritize_observed_proposals(case["previous"], case["current"], case["candidates"],
                                                        set(), attempted)
    assert ordered[0] == case["previous"][1]


def test_unused_observed_window_from_same_qualified_source_keeps_its_turn():
    candidate = {"id": "source", "source_url": "https://v.qq.com/delivery"}
    failed = {"candidate_id": "source", "clip_start": 31.98, "sampling_only": True}
    stairway = {"candidate_id": "source", "clip_start": 23.847, "sampling_only": True}
    next_window = {"candidate_id": "source-alias", "clip_start": 40.147, "sampling_only": True}
    candidates = {"source": candidate, "source-alias": candidate | {"id": "source-alias"}}
    attempted = {pooled_match.proposal_location(failed, candidates)}
    ordered = pooled_match.prioritize_observed_proposals([failed, stairway], [next_window],
                                                        candidates, {"source-alias"}, attempted)
    assert ordered == [stairway, next_window]
    assert not any({"approved", "verification", "verdict"} & p.keys() for p in ordered)
