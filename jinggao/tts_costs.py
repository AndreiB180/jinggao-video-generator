"""Account for speech at its submission boundary; never infer a cloud free tier.

Compatible endpoints require an explicitly configured, expiring tariff. Existing
Baidu account workflows remain usable without one, but their unpriced calls make
the complete cost and upper bound unknown. An auth token is not a price quote.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import json
import os
import sys

from . import costs

BAIDU = {"baidu_http", "baidu_web"}
UNITS = {"million_characters", "request"}


def validate_pricing(raw):
    """Validate a user-supplied RMB tariff; do not invent defaults or FX rates."""
    if not raw:
        return None
    try:
        value = json.loads(raw)
        required = {"provider", "base_url", "model", "voice", "unit", "cny_per_unit",
                    "billing", "valid_until", "evidence"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError
        if not all(isinstance(v, str) and v.strip() for v in value.values()):
            raise ValueError
        if value["provider"] not in BAIDU | {"compatible"} or value["unit"] not in UNITS:
            raise ValueError
        if value["billing"] not in {"metered", "included"}:
            raise ValueError
        rate = Decimal(value["cny_per_unit"])
        if not rate.is_finite() or not 0 <= rate <= 100000:
            raise ValueError
        if value["billing"] == "metered" and rate == 0:
            raise ValueError
        if value["billing"] == "included" and value["unit"] != "request":
            raise ValueError
        # The account routes have no confirmed character-metering protocol.
        if value["provider"] in BAIDU and value["unit"] != "request":
            raise ValueError
        if date.fromisoformat(value["valid_until"]).isoformat() != value["valid_until"]:
            raise ValueError
        return value
    except (ValueError, TypeError, InvalidOperation):
        raise ValueError("配音计费配置无效：须明确服务、地址、模型、音色、人民币单价、计量单位、计费依据和有效期；订阅权益须按次填写核实的分摊费用") from None


def identity(cfg):
    provider = cfg["tts_provider"]
    if provider in BAIDU:
        return "https://huibo.baidu.com", "public-voice", cfg["baidu_voice_name"].strip()
    if provider == "compatible":
        return cfg["tts_base_url"].rstrip("/"), cfg["tts_model"], cfg["tts_voice"]
    if provider == "system" and os.name == "nt":
        return "local://windows-sapi", "System.Speech", "installed-zh-CN"
    if provider == "system" and sys.platform == "darwin":
        return "local://macos-say", "say", "Tingting"
    raise ValueError("此配音方式没有可用的费用记录路径")


class speech_request(costs.request):
    def __init__(self, cfg, text, *, check=None):
        endpoint, model, voice = identity(cfg)
        payload = {"model": model, "voice": voice, "input": text, "response_format": "wav"}
        super().__init__(cfg | {"model_base_url": endpoint}, payload, "tts",
                         task="tts_" + cfg["tts_provider"], check=check)
        self.request_hash = costs.request_hash(endpoint + "/" + cfg["tts_provider"], payload)
        self.unit, self.usage_units = "request", 1
        self.amount = 0
        self.offline = cfg["tts_provider"] == "system"
        self.cost_basis = "offline_system" if self.offline else "unverified_account_entitlement"
        self.pricing_status = "known" if self.offline else "unpriced"
        if self.offline:
            # This is only the marginal API charge, never a claim about power,
            # equipment, voice quality or the acquisition cost of imported audio.
            self.rates = (Decimal(0),)
            return
        try:
            tariff = validate_pricing(cfg.get("tts_pricing_json", ""))
        except ValueError as exc:
            raise costs.PricingUnavailable(str(exc)) from None
        if tariff is None:
            if cfg["tts_provider"] in BAIDU:
                return
            raise costs.PricingUnavailable("兼容配音接口尚无已核实费率，已在发送前停止；请配置 tts_pricing_json 的人民币计价规则，不能把云配音默认记为免费")
        expected = (cfg["tts_provider"], endpoint, model, voice)
        actual = (tariff["provider"], tariff["base_url"].rstrip("/"), tariff["model"], tariff["voice"])
        if actual != expected:
            raise costs.PricingUnavailable("配音费率绑定的服务、地址、模型或音色与本次请求不符，已在发送前停止；请重新核实计费配置")
        if date.fromisoformat(tariff["valid_until"]) < date.fromisoformat(costs._day()):
            raise costs.PricingUnavailable("配音费率或订阅权益确认已过期，已在发送前停止；请核实额度、权益和有效期")
        self.unit = tariff["unit"]
        self.usage_units = len(text) if self.unit == "million_characters" else 1
        rate = Decimal(tariff["cny_per_unit"])
        self.amount = (int((rate * self.usage_units).to_integral_value(rounding=ROUND_CEILING))
                       if self.unit == "million_characters" else costs._micros(rate))
        self.rates = (rate,)
        self.cost_basis = "configured_" + tariff["billing"]
        self.pricing_status = "known"
        # Preserve the exact tariff's identity without persisting free-form
        # account evidence, credentials or original narration in the ledger.
        self.pricing_hash = costs.request_hash("tts-pricing-v1", tariff)

    def __enter__(self):
        if self.check:
            self.check()
        if self.offline or self.pricing_status == "unpriced":
            # Unknown money cannot truthfully pass a numeric cap. Preserve the
            # previously authorized account route and mark coverage incomplete.
            # Offline speech remains usable even on a historical over-cap project.
            with costs._db() as con:
                self._insert(con, 0)
            return self
        return self._reserve(self.amount)

    def success(self, code=None):
        self._close("unpriced" if self.pricing_status == "unpriced" else "settled", self.amount, code)

    def response(self, response):
        # Character/request-priced speech returns binary audio, not token usage.
        # Account on HTTP success even if later audio validation fails.
        if 200 <= response.status_code < 300:
            self.success(response.status_code)
        elif response.status_code in {400, 401, 402, 403, 404, 413, 422, 429}:
            self._close("rejected", 0, response.status_code)
        else:
            self._close("unknown", None, response.status_code)

    def __exit__(self, exc_type, exc, traceback):
        try:
            if not self.settled and self.sent and self.pricing_status == "unpriced":
                self._close("unpriced")
        finally:
            super().__exit__(exc_type, exc, traceback)
