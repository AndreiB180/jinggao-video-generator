import pytest

from jinggao import source_index, stock_queries


def test_all_40_topics_have_bounded_general_stock_seeds():
    assert set(stock_queries.TOPIC_QUERIES) == set(source_index.TOPICS)
    assert len(stock_queries.TOPIC_QUERIES) == 40
    for topic in source_index.TOPICS:
        for provider in ("pexels", "pixabay", "coverr", "commons"):
            terms = stock_queries.queries(topic, provider)
            assert 1 <= len(terms) <= 3
            assert len(terms) == len(set(terms))
            assert all(term.isascii() and term.strip() for term in terms)
    assert stock_queries.queries("不存在的类别", "pexels") == ()
    assert stock_queries.queries("农业乡村", "unknown") == ()


def test_nasa_only_uses_its_narrow_catalogue_domains():
    covered = {topic for topic in source_index.TOPICS if stock_queries.queries(topic, "nasa")}
    assert covered == {"自然环境", "科技互联网", "航天国防"}
    assert "rocket launch" in stock_queries.queries("航天国防", "nasa")
    for topic in ("美食烹饪", "时尚美妆", "名人动态", "历史档案"):
        assert stock_queries.queries(topic, "nasa") == ()


def test_rocket_launch_is_one_phrase_not_a_generic_subject_match():
    assert stock_queries.query_terms("火箭发射现场实拍") == ("rocket launch",)
    assert stock_queries.coverage("火箭发射", "A rocket launch from the pad") == 1
    assert stock_queries.coverage("火箭发射", "a rocket on display") == 0
    assert stock_queries.coverage("火箭发射", "the launch of a phone beside a model rocket") == 0


def test_tea_spreading_is_not_tea_picking_or_generic_tea():
    assert stock_queries.query_terms("茶叶摊放") == ("spreading tea leaves",)
    assert stock_queries.query_terms("摊放鲜叶实拍") == ("spreading tea leaves",)
    assert stock_queries.query_terms("采茶") == ("tea picking",)
    assert stock_queries.coverage("茶叶摊放", "Workers spreading tea leaves on trays") == 1
    assert stock_queries.coverage("茶叶摊放", "Workers picking tea leaves in a tea plantation") == 0
    assert stock_queries.coverage("茶叶摊放", "a cup of tea") == 0
    assert stock_queries.coverage("采茶", "spreading tea leaves") == 0


@pytest.mark.parametrize("query", ["张云涛", "马龙", "李泽昱采访", "Unknown Person"])
def test_unknown_names_are_not_translated(query):
    assert stock_queries.query_terms(query) == ()
    assert stock_queries.query_profile(query)["original"] == query
    assert stock_queries.coverage(query, "an interview with a person") == 0


@pytest.mark.parametrize("query, title", [
    ("马龙机场", "airport terminal"),
    ("火箭发射失败现场", "rocket launch footage"),
    ("2024年火箭发射", "rocket launch"),
    ("中国火箭发射", "rocket launch"),
    ("不是采茶", "tea picking"),
    ("NASA rocket launch failure", "rocket launch"),
])
def test_unknown_constraints_cannot_be_erased_by_a_mapped_phrase(query, title):
    profile = stock_queries.query_profile(query)
    assert profile["queries"] and profile["unmapped"] and not profile["complete"]
    assert profile["original"] == query
    assert stock_queries.coverage(query, title) == 0


@pytest.mark.parametrize("query, title", [
    ("火箭", "rocketry"),
    ("火箭发射", "rocket launcher"),
    ("茶叶", "tea leaflets"),
    ("机场", "myairport terminal"),
    ("机器人", "robotic machinery"),
])
def test_english_matches_require_exact_word_boundaries(query, title):
    assert stock_queries.coverage(query, title) == 0


def test_known_aliases_and_phrase_separators_are_explicit():
    assert stock_queries.coverage("火箭发射", "ROCKET-LAUNCH footage") == 1
    assert stock_queries.coverage("采茶", "plucking tea leaves") == 1
    assert stock_queries.coverage("数据中心", "a data centre") == 1
    assert stock_queries.query_terms("launching rockets") == ("rocket launch",)


def test_topic_seeds_do_not_become_live_query_evidence():
    assert stock_queries.queries("农业乡村", "pexels")
    assert stock_queries.query_terms("农业乡村") == ()
    assert stock_queries.coverage("农业乡村", "tea picking") == 0


def test_output_is_bounded_without_truncating_coverage_requirements():
    query = "火箭发射、空间站、航天员训练、卫星发射、地球观测"
    profile = stock_queries.query_profile(query)
    assert profile["complete"] and len(profile["queries"]) == 4
    assert len(profile["concepts"]) == 5
    assert stock_queries.coverage(query, "rocket launch") == .2
    assert stock_queries.coverage(query, "rocket launch, space station, astronaut training, satellite launch, Earth observation") == 1


def test_empty_or_only_query_wrappers_do_not_gain_coverage():
    for query in ("", "实拍视频素材", "4k footage"):
        assert stock_queries.query_terms(query) == ()
        assert stock_queries.coverage(query, "rocket launch tea picking") == 0
