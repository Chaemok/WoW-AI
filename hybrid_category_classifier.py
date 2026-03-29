from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_GMS_OPENAI_BASE_URL = "https://gms.ssafy.io/gmsapi/api.openai.com/v1"
EXCLUDE_LABEL = "제외"


@dataclass(slots=True)
class CategoryDecision:
    category: str
    confidence: float
    reason: str
    provider: str


class GMSCategoryClassifier:
    def __init__(self, allowed_categories: list[str]) -> None:
        self.allowed_categories = sorted(set(allowed_categories))
        self.enabled = os.environ.get("CATEGORY_LLM_ENABLED", "0") == "1"
        self.base_url = os.environ.get(
            "CATEGORY_LLM_BASE_URL",
            DEFAULT_GMS_OPENAI_BASE_URL,
        ).rstrip("/")
        self.model = os.environ.get("CATEGORY_LLM_MODEL", "gpt-5.2")
        self.timeout_sec = int(os.environ.get("CATEGORY_LLM_TIMEOUT_SEC", "20"))
        self.api_key = (
            os.environ.get("CATEGORY_LLM_API_KEY")
            or os.environ.get("GMS_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()

    @property
    def ready(self) -> bool:
        return self.enabled and bool(self.api_key)

    def classify(
        self,
        *,
        merchant_name: str,
        transaction_detail: str,
        payment_method: str,
        amount: int,
    ) -> CategoryDecision | None:
        if not self.ready:
            return None

        request_body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": self._build_system_prompt(),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "merchant_name": merchant_name,
                            "transaction_detail": transaction_detail,
                            "payment_method": payment_method,
                            "amount": amount,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request_body,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
        except requests.RequestException:
            return None

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            return None

        category = str(parsed.get("category") or "").strip()
        if category not in self.allowed_categories and category != EXCLUDE_LABEL:
            return None

        try:
            confidence = float(parsed.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.0

        reason = str(parsed.get("reason") or "").strip()
        if not reason:
            reason = "LLM classified the merchant from the available transaction fields."

        return CategoryDecision(
            category=category or EXCLUDE_LABEL,
            confidence=max(0.0, min(confidence, 1.0)),
            reason=reason,
            provider=self.model,
        )

    def classify_many(self, transactions: list[dict]) -> dict[str, CategoryDecision]:
        if not self.ready or not transactions:
            return {}

        payload = []
        merchant_order: list[str] = []
        for transaction in transactions:
            merchant_name = str(transaction.get("merchant_name") or "").strip()
            if not merchant_name or merchant_name in merchant_order:
                continue
            merchant_order.append(merchant_name)
            payload.append(
                {
                    "merchant_name": merchant_name,
                    "transaction_detail": str(transaction.get("transaction_detail") or "").strip(),
                    "payment_method": str(transaction.get("payment_method") or "").strip(),
                    "amount": int(transaction.get("amount") or 0),
                }
            )

        if not payload:
            return {}

        request_body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": self._build_batch_system_prompt(),
                },
                {
                    "role": "user",
                    "content": json.dumps({"transactions": payload}, ensure_ascii=False),
                },
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request_body,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
        except requests.RequestException:
            return {}

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            raw_results = parsed["results"]
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            return {}

        decisions: dict[str, CategoryDecision] = {}
        if not isinstance(raw_results, list):
            return decisions

        for item in raw_results:
            if not isinstance(item, dict):
                continue
            merchant_name = str(item.get("merchant_name") or "").strip()
            category = str(item.get("category") or "").strip()
            if not merchant_name or not category:
                continue
            if category not in self.allowed_categories and category != EXCLUDE_LABEL:
                continue
            try:
                confidence = float(item.get("confidence"))
            except (TypeError, ValueError):
                confidence = 0.0
            reason = str(item.get("reason") or "").strip()
            if not reason:
                reason = "LLM classified the merchant from the available transaction fields."
            decisions[merchant_name] = CategoryDecision(
                category=category,
                confidence=max(0.0, min(confidence, 1.0)),
                reason=reason,
                provider=self.model,
            )

        return decisions

    def _build_system_prompt(self) -> str:
        categories = "\n".join(f"- {category}" for category in self.allowed_categories)
        return (
            "You classify one transaction into exactly one allowed category.\n"
            "Use merchant_name as the strongest signal.\n"
            "Use transaction_detail, payment_method, and amount only as supporting hints.\n"
            "If the merchant is too ambiguous, return '제외'.\n"
            "Do not invent new categories.\n\n"
            "[Allowed categories]\n"
            f"{categories}\n"
            f"- {EXCLUDE_LABEL}\n\n"
            "Return JSON only.\n"
            '{"category":"...", "confidence":0.0, "reason":"..."}'
        )

    def _build_batch_system_prompt(self) -> str:
        categories = "\n".join(f"- {category}" for category in self.allowed_categories)
        return (
            "You classify multiple transactions.\n"
            "Use merchant_name as the strongest signal.\n"
            "Use transaction_detail, payment_method, and amount only as supporting hints.\n"
            "Return the best matching allowed category for each merchant.\n"
            "Only return '제외' when the merchant is truly too ambiguous.\n"
            "Do not invent new categories.\n\n"
            "[Allowed categories]\n"
            f"{categories}\n"
            f"- {EXCLUDE_LABEL}\n\n"
            "Return JSON only.\n"
            '{"results":[{"merchant_name":"...", "category":"...", "confidence":0.0, "reason":"..."}]}'
        )


EXCLUDE_LABEL = "제외"


def _patched_build_system_prompt(self: GMSCategoryClassifier) -> str:
    categories = "\n".join(f"- {category}" for category in self.allowed_categories)
    return (
        "You classify one transaction into exactly one allowed category.\n"
        "Use merchant_name as the strongest signal.\n"
        "Use transaction_detail, payment_method, and amount only as supporting hints.\n"
        "If the merchant is too ambiguous, return '제외'.\n"
        "Do not invent new categories.\n\n"
        "[Allowed categories]\n"
        f"{categories}\n"
        f"- {EXCLUDE_LABEL}\n\n"
        "Return JSON only.\n"
        '{"category":"...", "confidence":0.0, "reason":"..."}'
    )


GMSCategoryClassifier._build_system_prompt = _patched_build_system_prompt
