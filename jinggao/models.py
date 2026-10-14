from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NewProject(Strict):
    title: str = Field(default="未命名项目", min_length=1, max_length=100)
    script: str = Field(default="", max_length=30000)
    aspect: Literal["16:9", "9:16"] = "16:9"
    media_policy: Literal["video_preferred", "video_only", "mixed"] = "video_preferred"
    min_video_ratio: float = Field(default=0.8, ge=0, le=1)
    min_video_height: Literal[270, 480, 720, 1080] = 720


class ProjectPatch(Strict):
    title: str | None = Field(default=None, min_length=1, max_length=100)
    script: str | None = Field(default=None, max_length=30000)
    aspect: Literal["16:9", "9:16"] | None = None
    style: str | None = Field(default=None, max_length=500)
    requirements: str | None = Field(default=None, max_length=2000)
    media_policy: Literal["video_preferred", "video_only", "mixed"] | None = None
    min_video_ratio: float | None = Field(default=None, ge=0, le=1)
    min_video_height: Literal[270, 480, 720, 1080] | None = None


class RewriteRequest(Strict):
    text: str = Field(min_length=1, max_length=30000)
    title: str = Field(default="", max_length=200)
    minutes: float | None = Field(default=None, ge=0.5, le=15)
    aspect: Literal["16:9", "9:16"] = "16:9"
    project_id: str | None = Field(default=None, max_length=64)


class NewsLink(Strict):
    url: str = Field(min_length=8, max_length=2000)


class NewsText(Strict):
    title: str = Field(default="", max_length=200)
    text: str = Field(min_length=30, max_length=30000)
    source: str = Field(default="", max_length=80)


class NewsCrawl(Strict):
    koubo: bool = False


class NewsProduce(Strict):
    ids: list[Annotated[str, Field(pattern=r"^[a-f0-9]{16}$")]] = Field(min_length=1, max_length=20)
    aspect: Literal["16:9", "9:16"] = "16:9"
    minutes: float | None = Field(default=None, ge=0.5, le=15)
    resolution: Literal["720p", "1080p"] = "720p"
    auto: bool = True


class PersonTarget(Strict):
    name: str = Field(min_length=2, max_length=100)
    aliases: list[Annotated[str, Field(min_length=2, max_length=100)]] = Field(default_factory=list, max_length=8)


class ShotPatch(Strict):
    narration: str | None = Field(default=None, min_length=1, max_length=2000)
    visual: str | None = Field(default=None, max_length=2000)
    query: str | None = Field(default=None, max_length=400)
    subject: str | None = Field(default=None, max_length=300)
    location: str | None = Field(default=None, max_length=200)
    era: str | None = Field(default=None, max_length=200)
    exclusions: str | None = Field(default=None, max_length=500)
    overlay_label: str | None = Field(default=None, max_length=30)
    duration: float | None = Field(default=None, ge=1, le=120)
    clip_start: float | None = Field(default=None, ge=0, le=86400)
    locked: bool | None = None
    approved: bool | None = None
    fit: Literal["contain", "cover"] | None = None
    persons: list[PersonTarget] | None = Field(default=None, max_length=8)
    event: str | None = Field(default=None, max_length=400)
    evidence_mode: Literal["person", "event", "generic"] | None = None
    event_scope: Literal["exact", "background"] | None = None
    preferred_media: Literal["video", "image", "any"] | None = None
    source_review_note: str | None = Field(default=None, max_length=1000)
    source_review_url: str | None = Field(default=None, max_length=2000)


class JobRequest(Strict):
    kind: Literal["plan", "search", "match", "preview", "select", "voice", "voice_full", "render", "verify", "produce"]
    shot_id: str | None = None
    shot_ids: list[str] | None = Field(default=None, min_length=1, max_length=160)
    candidate_id: str | None = None
    mode: Literal["auto", "rules", "model"] = "auto"
    rewrite: bool = False
    replace: bool = False
    allow_silent: bool = False
    resolution: Literal["720p", "1080p"] = "1080p"
    music_volume: float = Field(default=0.12, ge=0, le=0.5)
    image_motion: bool = True
    clip_start: float | None = Field(default=None, ge=0, le=86400)


class BatchItem(NewProject):
    script: str = Field(min_length=1, max_length=30000)

    @field_validator("script", "title")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("文稿与标题不能为空白")
        return value.strip()


class BatchRequest(Strict):
    projects: list[BatchItem] = Field(min_length=1, max_length=20)
    requirements: str = Field(default="", max_length=2000)
    resolution: Literal["720p", "1080p"] = "720p"


class AssetPatch(Strict):
    title: str | None = Field(default=None, max_length=300)
    description: str | None = Field(default=None, max_length=1000)
    source_url: str | None = Field(default=None, max_length=2000)
    license: str | None = Field(default=None, max_length=500)
    license_url: str | None = Field(default=None, max_length=2000)
    author: str | None = Field(default=None, max_length=300)


class RemoteImport(Strict):
    url: str = Field(max_length=4000)
    title: str = Field(default="链接素材", max_length=300)
    source_url: str = Field(default="", max_length=2000)
    license: str = Field(default="待确认", max_length=500)


class LibraryBuild(Strict):
    action: Literal["collect", "rebuild", "index"] = "collect"
    queries: list[Annotated[str, Field(min_length=2, max_length=100)]] = Field(default_factory=list, max_length=4)
    urls: list[Annotated[str, Field(min_length=10, max_length=2000)]] = Field(default_factory=list, max_length=12)
    max_sources: int = Field(default=6, ge=1, le=12)
    clips_per_source: int = Field(default=2, ge=1, le=2)
    clip_seconds: int = Field(default=12, ge=4, le=30)
    index_target: int = Field(default=50000, ge=1000, le=500000)
    index_topics: list[Annotated[str, Field(max_length=30)]] = Field(default_factory=list, max_length=60)
    index_entities: bool = True
    index_providers: list[Literal["tencent", "bilibili", "cctv", "chinanews", "iqiyi", "sohu", "youku", "baidu_video",
                                  "pexels", "pixabay", "coverr", "commons", "nasa"]] = Field(default_factory=list, max_length=13)


class LibraryUse(Strict):
    project_id: str = Field(pattern=r"^[a-f0-9]{32}$")
