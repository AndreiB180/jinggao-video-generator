"""Named records for joint pixel observations, preserving all original checks.

The provider's image JSON mode does not enforce a schema. Fixed-position rows
proved fragile in actual replies, so each frame now carries explicit fields.
No decoder supplies omitted observations or moves fields between frames/groups.
"""
from __future__ import annotations

from copy import deepcopy
import json

from . import pixel_results

VERSION = "joint_pixel_records_v2"
FRAME_COLUMNS = ("frame_index", "objects", "action", "text", "representation", "type_evidence", "overlays")
OVERLAY_COLUMNS = ("kind", "text", "bbox", "impact", "occludes")
PROTOCOL = """
逐帧使用命名字段的JSON对象，不使用位置数组。所有字段必填，不用省略项、null或前帧引用代替。
每个frames元素包含frame_index、objects、action、text、representation、type_evidence、overlays七个字段。
每个overlays元素包含kind、text、bbox、impact、occludes五个字段。没有覆盖物为[]，没有可读文字text=[]。
representation、kind、impact使用上述完整英文枚举；bbox仍是完整图片0–1坐标[左,上,右,下]，不改变单位。
输入observation_groups列出完整批次。每个group_id恰好一次，每组所有frame_index各恰好一次；不串组、不借用别帧。
按输入组顺序逐组填写，当前组的全部帧写入它自己的frames数组后，再写下一组。组外不放置任何帧。
只返回一个JSON对象，schema必须逐字为joint_pixel_records_v2；如下仅演示一帧的字段，实际返回所有输入帧：
{"schema":"joint_pixel_records_v2","groups":[{"id":"输入group_id","scene_description":"本组中立概括",
"frames":[{"frame_index":0,"objects":"实际主体物体环境","action":"实际动作或姿态","text":["可读原文"],
"representation":"live_action","type_evidence":"像素依据",
"overlays":[{"kind":"caption","text":"可读字","bbox":[0,0.8,1,1],"impact":"minor","occludes":""}]}]}]}。
不返回其他字段，不输出审批、匹配、真实性或人物身份结论。
"""


def pack(body):
    """Encode a canonical body without supplying missing fields or approvals."""
    if not isinstance(body, dict) or set(body) != {"groups"} or not isinstance(body["groups"], list):
        raise ValueError("联合观察编码需要完整分组")
    result = {"schema": VERSION, "groups": []}
    for group in body["groups"]:
        if (not isinstance(group, dict) or set(group) != {"id", "scene_description", "frames"}
                or not isinstance(group["frames"], list)):
            raise ValueError("联合观察编码缺少分组字段")
        for frame in group["frames"]:
            if not isinstance(frame, dict) or set(frame) != set(FRAME_COLUMNS):
                raise ValueError("联合观察编码缺少逐帧字段")
            overlays = frame["overlays"]
            if (not isinstance(overlays, list) or any(not isinstance(row, dict)
                    or set(row) != set(OVERLAY_COLUMNS) for row in overlays)):
                raise ValueError("联合观察编码缺少覆盖物字段")
        result["groups"].append(deepcopy(group))
    return result


def expand(raw):
    """Decode complete named records; never infer or repair a field."""
    try:
        body = json.loads(raw, object_pairs_hook=pixel_results.unique_object)
    except (TypeError, ValueError):
        raise ValueError("联合观察未返回无歧义的完整JSON") from None
    if (not isinstance(body, dict) or set(body) != {"schema", "groups"}
            or body["schema"] != VERSION or not isinstance(body["groups"], list)):
        raise ValueError("联合观察记录协议或字段无效")
    # pack validates every record's fields without synthesizing missing data.
    # The caller still checks exact group/frame IDs, types, enums and evidence.
    return {"groups": pack({"groups": body["groups"]})["groups"]}
