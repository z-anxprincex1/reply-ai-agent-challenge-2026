from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from math import asin, ceil, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import ulid
from dotenv import load_dotenv
from langfuse import Langfuse


TRANSACTION_ALIASES: dict[str, tuple[str, ...]] = {
    "transaction_id": ("transaction_id", "tx_id", "id"),
    "sender_id": ("sender_id", "source_user_id", "origin_user_id"),
    "recipient_id": ("recipient_id", "target_user_id", "destination_user_id"),
    "transaction_type": ("transaction_type", "type", "category"),
    "amount": ("amount", "value", "transaction_amount"),
    "location": ("location", "merchant_location", "city"),
    "payment_method": ("payment_method", "channel", "instrument"),
    "sender_iban": ("sender_iban", "source_iban", "origin_iban"),
    "recipient_iban": ("recipient_iban", "target_iban", "destination_iban"),
    "balance_after": ("balance_after", "post_balance", "available_balance"),
    "description": ("description", "memo", "note"),
    "timestamp": ("timestamp", "created_at", "datetime", "date"),
}

URL_PATTERN = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
BARE_URL_PATTERN = re.compile(r"\b(?:bit\.ly|tinyurl\.com|[a-z0-9-]+\.(?:com|net|org|de|it|co\.uk))(?:/[^\s'\"<>]*)?", re.IGNORECASE)
MAIL_DATE_PATTERN = re.compile(r"^Date:\s*(.+)$", re.MULTILINE)
SMS_DATE_PATTERN = re.compile(r"Date:\s*([0-9:\-\s]+)")
URGENCY_PATTERN = re.compile(
    r"\b(urgent|alert|verify|locked|lockout|suspend|suspension|security|confirm|login|sign-?in)\b",
    re.IGNORECASE,
)
SENSITIVE_BRAND_PATTERN = re.compile(
    r"\b(paypal|amazon|social security|ssa|pension|bank of america|chase|barclays|commerzbank|unicredit|netflix|uber|card|benefit)\b",
    re.IGNORECASE,
)
PAYMENT_ACTION_PATTERN = re.compile(
    r"\b(pay(?:ment)?|release fee|claim|verify(?: your)? identity|confirm(?: your)? (?:account|details)|login|sign-?in|password|credential|bank account|customs fee|suspend(?:ed|ion)?)\b",
    re.IGNORECASE,
)
BENIGN_COMMUNICATION_PATTERN = re.compile(
    r"\b(rsvp|webinar|workshop|forum|networking|meeting|benefits update|order confirmation|receipt|track package|help center|unsubscribe|reply stop|free business support|training schedule|regional briefing|invitation|student services|library|campus|orientation fair|exam prep)\b",
    re.IGNORECASE,
)
SAFETY_ASSURANCE_PATTERN = re.compile(
    r"never ask for (?:passwords?|bank account details|sensitive credentials)|did not place this order|privacy note",
    re.IGNORECASE,
)
TRAINING_SIMULATION_PATTERN = re.compile(
    r"\b(training (?:simulation|exercise)|simulation notice|phishing-awareness|security awareness|anti-phishing training|simulated phishing|for training purposes)\b",
    re.IGNORECASE,
)
LEGITIMATE_OBLIGATION_PATTERN = re.compile(
    r"\b(?:rent|lease|mortgage|loan payment|salary|payroll|insurance|premium|utility bill|water bill|gas bill|electricity|phone bill|subscription|monthly fee|membership|tuition|supplier payment|invoice settlement|office supplies|savings transfer|emergency fund transfer|donation)\b",
    re.IGNORECASE,
)
TRUSTED_DOMAIN_SUFFIXES = (
    "gov.uk",
    "dresden.de",
    "novaworks.com",
    "charleston-consulting.com",
    "workwise-consulting.com",
    "consultinghub.com",
    "marketlane.com",
    "business-advisory.co.uk",
    "fedex.com",
    "amazon.com",
    "amazon.co.uk",
    "paypal.com",
    "linkedin.com",
    "eventbrite.com",
    "zoom.us",
    "google.com",
    "calendar.google.com",
    "amtrak.com",
    "uber.com",
)

DATASET_ID_ALIASES = {
    "l1": "l1",
    "the-truman-show": "l1",
    "the truman show": "l1",
    "truman-show": "l1",
    "l2": "l2",
    "deus-ex": "l2",
    "deus ex": "l2",
    "l3": "l3",
    "brave-new-world": "l3",
    "brave new world": "l3",
    "l4": "l4",
    "blade-runner": "l4",
    "blade runner": "l4",
    "l5": "l5",
    "1984": "l5",
}


@dataclass(slots=True)
class UserProfile:
    user_key: str
    first_name: str
    last_name: str
    full_name: str
    salary: float | None
    iban: str | None
    residence_city: str | None
    residence_lat: float | None
    residence_lng: float | None
    vulnerability_score: float
    biotag: str | None = None


@dataclass(slots=True)
class DatasetContext:
    dataset_name: str
    base_path: Path
    transactions: pd.DataFrame
    users: list[UserProfile]
    sms_messages: list[dict[str, Any]]
    mail_messages: list[dict[str, Any]]
    audio_events: list[dict[str, Any]]
    locations: pd.DataFrame
    tx_cols: dict[str, str | None]
    user_by_iban: dict[str, UserProfile]
    user_by_biotag: dict[str, UserProfile]


@dataclass(slots=True)
class AgentResult:
    name: str
    frame: pd.DataFrame


def normalize_key(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def canonical_dataset_id(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    return DATASET_ID_ALIASES.get(normalized, normalized)


def infer_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {normalize_key(column): column for column in columns}
    for alias in aliases:
        alias_key = normalize_key(alias)
        if alias_key in normalized:
            return normalized[alias_key]
    for column in columns:
        column_key = normalize_key(column)
        if any(normalize_key(alias) in column_key or column_key in normalize_key(alias) for alias in aliases):
            return column
    return None


def load_json(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and len(payload) == 1:
        only_value = next(iter(payload.values()))
        if isinstance(only_value, list):
            return only_value
    return payload


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "null_counts": {column: int(value) for column, value in df.isna().sum().items()},
        "sample": df.head(3).replace({pd.NA: None}).to_dict(orient="records"),
    }


def summarize_json(payload: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"top_level_type": type(payload).__name__}
    if hasattr(payload, "__len__"):
        summary["length"] = int(len(payload))
    if isinstance(payload, list) and payload:
        summary["keys"] = list(payload[0].keys()) if isinstance(payload[0], dict) else []
        summary["sample"] = payload[:2]
    return summary


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return pd.NaT
    return timestamp.tz_convert(None)


def load_project_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path, override=True)


def resolve_team_name() -> str:
    load_project_env()
    return os.getenv("TEAM_NAME", "reply-agentic").replace(" ", "-")


def generate_session_id() -> str:
    return f"{resolve_team_name()}-{ulid.new().str.lower()}"


def build_langfuse() -> Langfuse | None:
    load_project_env()
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")
    if not (public_key and secret_key and host):
        return None
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def normalize_name_token(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z]", "", value.lower())


def load_audio_events(base_path: Path, users: list[UserProfile]) -> list[dict[str, Any]]:
    audio_dir = base_path / "audio"
    if not audio_dir.exists() or not audio_dir.is_dir():
        return []

    user_lookup = {normalize_name_token(user.full_name): user for user in users if user.full_name}
    events: list[dict[str, Any]] = []
    for file_path in sorted(audio_dir.glob("*.mp3")):
        stem = file_path.stem
        if "-" not in stem:
            continue
        timestamp_raw, speaker_raw = stem.split("-", 1)
        timestamp = pd.to_datetime(timestamp_raw, format="%Y%m%d_%H%M%S", errors="coerce")
        if pd.isna(timestamp):
            continue
        speaker_key = normalize_name_token(speaker_raw)
        user = user_lookup.get(speaker_key)
        events.append(
            {
                "timestamp": timestamp,
                "speaker_raw": speaker_raw,
                "user_key": user.user_key if user else None,
                "file_name": file_path.name,
            }
        )
    return events


def infer_biotags(users: list[UserProfile], locations: pd.DataFrame) -> dict[str, UserProfile]:
    if locations.empty:
        return {}

    grouped = (
        locations.dropna(subset=["biotag"])
        .groupby("biotag", dropna=True)
        .agg(home_city=("city", lambda values: values.mode().iloc[0] if not values.mode().empty else None))
        .reset_index()
    )
    remaining = grouped.to_dict(orient="records")
    mapping: dict[str, UserProfile] = {}

    for user in users:
        match = None
        for candidate in remaining:
            if user.residence_city and candidate.get("home_city") and str(candidate["home_city"]).lower() == user.residence_city.lower():
                match = candidate
                break
        if match is None and remaining:
            match = remaining[0]
        if match is not None:
            user.biotag = str(match["biotag"])
            mapping[user.biotag] = user
            remaining = [candidate for candidate in remaining if candidate["biotag"] != user.biotag]
    return mapping


def infer_vulnerability(description: str | None, birth_year: Any, job: str | None) -> float:
    score = 0.0
    text = (description or "").lower()
    if job and "retired" in job.lower():
        score += 0.18
    try:
        year = int(birth_year) if birth_year not in (None, "") else None
    except (TypeError, ValueError):
        year = None
    if year is not None and year <= 2015:
        score += 0.15
    for phrase, weight in (
        ("phishing", 0.18),
        ("tende a fidarsi", 0.16),
        ("not immune", 0.12),
        ("flashy online lure", 0.12),
        ("probability", 0.10),
        ("pièges", 0.10),
        ("trappole", 0.10),
        ("cautious", -0.05),
    ):
        if phrase in text:
            score += weight
    percent_match = re.search(r"(\d{1,2})\s*%", text)
    if percent_match:
        score += min(int(percent_match.group(1)) / 100.0, 0.35)
    return max(0.0, min(score, 0.65))


def load_context(base_path: Path, dataset_name: str) -> DatasetContext:
    transactions = pd.read_csv(base_path / "transactions.csv")
    tx_cols = {name: infer_column(list(transactions.columns), aliases) for name, aliases in TRANSACTION_ALIASES.items()}

    for canonical in ("timestamp", "amount", "balance_after"):
        source = tx_cols.get(canonical)
        target = f"_{canonical}"
        if source:
            if canonical == "timestamp":
                transactions[target] = pd.to_datetime(transactions[source], errors="coerce")
            else:
                transactions[target] = pd.to_numeric(transactions[source], errors="coerce")
        else:
            transactions[target] = pd.NA

    for canonical in ("sender_iban", "recipient_iban", "location", "payment_method", "description", "sender_id", "recipient_id", "transaction_type"):
        source = tx_cols.get(canonical)
        target = f"_{canonical}"
        transactions[target] = transactions[source].fillna("").astype(str).str.strip() if source else ""
        if "iban" in canonical:
            transactions[target] = transactions[target].str.upper()

    users_payload = load_json(base_path / "users.json")
    sms_payload = load_json(base_path / "sms.json")
    mail_payload = load_json(base_path / "mails.json")
    location_payload = load_json(base_path / "locations.json")

    users: list[UserProfile] = []
    for index, row in enumerate(users_payload if isinstance(users_payload, list) else [], start=1):
        residence = row.get("residence") or {}
        first_name = str(row.get("first_name") or "").strip()
        last_name = str(row.get("last_name") or "").strip()
        job = str(row.get("job")).strip() if row.get("job") else None
        description = str(row.get("description") or "")
        users.append(
            UserProfile(
                user_key=f"user_{index}",
                first_name=first_name,
                last_name=last_name,
                full_name=" ".join(part for part in (first_name, last_name) if part).strip(),
                salary=safe_float(row.get("salary")),
                iban=str(row.get("iban")).strip().upper() if row.get("iban") else None,
                residence_city=str(residence.get("city")).strip() if residence.get("city") else None,
                residence_lat=safe_float(residence.get("lat")),
                residence_lng=safe_float(residence.get("lng")),
                vulnerability_score=infer_vulnerability(description, row.get("birth_year"), job),
            )
        )

    locations = pd.DataFrame(location_payload if isinstance(location_payload, list) else [])
    if not locations.empty:
        if "timestamp" in locations.columns:
            locations["timestamp"] = pd.to_datetime(locations["timestamp"], errors="coerce")
        for column in ("lat", "lng"):
            if column in locations.columns:
                locations[column] = pd.to_numeric(locations[column], errors="coerce")
        for column in ("city", "biotag"):
            if column in locations.columns:
                locations[column] = locations[column].fillna("").astype(str).str.strip()

    user_by_iban = {user.iban: user for user in users if user.iban}
    user_by_biotag = infer_biotags(users, locations)
    audio_events = load_audio_events(base_path, users)

    print(f"Dataset: {base_path.resolve()}")
    tx_summary = summarize_dataframe(transactions)
    print("transactions.csv")
    print(f"  rows: {tx_summary['rows']}")
    print(f"  columns: {tx_summary['columns']}")
    print(f"  inferred schema: {tx_cols}")
    print(f"  null counts: {tx_summary['null_counts']}")
    for sample in tx_summary["sample"]:
        print(f"  sample: {sample}")

    for name, payload in {
        "users.json": users_payload,
        "sms.json": sms_payload,
        "mails.json": mail_payload,
        "locations.json": location_payload,
    }.items():
        summary = summarize_json(payload)
        print(name)
        print(f"  summary: {summary}")

    return DatasetContext(
        dataset_name=dataset_name,
        base_path=base_path,
        transactions=transactions,
        users=users,
        sms_messages=sms_payload if isinstance(sms_payload, list) else [],
        mail_messages=mail_payload if isinstance(mail_payload, list) else [],
        audio_events=audio_events,
        locations=locations,
        tx_cols=tx_cols,
        user_by_iban=user_by_iban,
        user_by_biotag=user_by_biotag,
    )


class BaseAgent:
    name = "base"

    def empty_result(self, context: DatasetContext) -> pd.DataFrame:
        tx_id_col = context.tx_cols["transaction_id"]
        return pd.DataFrame(
            {
                "transaction_id": context.transactions[tx_id_col],
                "score": 0.0,
                "reasons": [[] for _ in range(len(context.transactions))],
            }
        )


class TransactionPatternAgent(BaseAgent):
    name = "transaction_pattern"

    def analyze(self, context: DatasetContext) -> AgentResult:
        result = self.empty_result(context)
        tx = context.transactions.sort_values("_timestamp", kind="stable")
        tx_id_col = context.tx_cols["transaction_id"]
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, list[str]] = defaultdict(list)
        global_q90 = tx["_amount"].quantile(0.90)
        global_q97 = tx["_amount"].quantile(0.97)

        for _, sender_df in tx.groupby("_sender_iban", dropna=False):
            sender_df = sender_df.reset_index(drop=True)
            q75 = sender_df["_amount"].quantile(0.75)
            q25 = sender_df["_amount"].quantile(0.25)
            iqr = max(q75 - q25, 1.0)
            sender_median = sender_df["_amount"].median()
            seen_recipients: set[str] = set()

            for _, row in sender_df.iterrows():
                transaction_id = row[tx_id_col]
                amount = row["_amount"]
                timestamp = row["_timestamp"]
                recipient_iban = row["_recipient_iban"]

                if pd.notna(amount) and amount > max(global_q97, q75 + 2.5 * iqr):
                    scores[transaction_id] += 0.5
                    reasons[transaction_id].append("amount is an extreme outlier for this sender")
                elif pd.notna(amount) and amount > max(global_q90, q75 + 1.5 * iqr):
                    scores[transaction_id] += 0.3
                    reasons[transaction_id].append("amount is unusually high for this sender")

                if recipient_iban and recipient_iban not in seen_recipients and pd.notna(amount) and amount > max(sender_median * 1.3, global_q90):
                    scores[transaction_id] += 0.2
                    reasons[transaction_id].append("first transfer to this recipient is unexpectedly large")
                if recipient_iban:
                    seen_recipients.add(recipient_iban)

                if pd.notna(timestamp):
                    recent = sender_df.loc[
                        (sender_df["_timestamp"] <= timestamp) & (sender_df["_timestamp"] >= timestamp - pd.Timedelta(hours=24))
                    ]
                    if len(recent) >= 4 and recent["_amount"].sum() > max(sender_median * 3, global_q90 * 1.5):
                        scores[transaction_id] += 0.25
                        reasons[transaction_id].append("part of a dense 24-hour transaction burst")

                balance_after = row["_balance_after"]
                if pd.notna(amount) and pd.notna(balance_after):
                    balance_before = amount + balance_after
                    if balance_before > 0 and amount / balance_before > 0.65:
                        scores[transaction_id] += 0.15
                        reasons[transaction_id].append("uses most of the apparent account balance")

        result["score"] = result["transaction_id"].map(scores).fillna(0.0).clip(0.0, 1.0)
        result["reasons"] = result["transaction_id"].map(lambda transaction_id: reasons.get(transaction_id, []))
        return AgentResult(self.name, result)


class UserBehaviorAgent(BaseAgent):
    name = "user_behavior"

    def resolve_user(self, row: pd.Series, context: DatasetContext) -> UserProfile | None:
        sender_iban = row["_sender_iban"]
        sender_id = row["_sender_id"]
        if sender_iban and sender_iban in context.user_by_iban:
            return context.user_by_iban[sender_iban]
        if sender_id and sender_id in context.user_by_biotag:
            return context.user_by_biotag[sender_id]
        return None

    def analyze(self, context: DatasetContext) -> AgentResult:
        result = self.empty_result(context)
        tx = context.transactions.sort_values("_timestamp", kind="stable")
        tx_id_col = context.tx_cols["transaction_id"]
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, list[str]] = defaultdict(list)
        hour_windows: dict[str, tuple[float, float]] = {}

        for sender_iban, sender_df in tx.groupby("_sender_iban", dropna=False):
            hours = sender_df["_timestamp"].dropna().dt.hour
            if len(hours) >= 5:
                hour_windows[str(sender_iban)] = (float(hours.quantile(0.10)), float(hours.quantile(0.90)))

        for _, row in tx.iterrows():
            transaction_id = row[tx_id_col]
            user = self.resolve_user(row, context)
            amount = row["_amount"]
            timestamp = row["_timestamp"]

            if user and user.salary and pd.notna(amount):
                monthly_salary = user.salary / 12.0
                if amount > monthly_salary * 1.5:
                    scores[transaction_id] += 0.55
                    reasons[transaction_id].append("transaction exceeds 150% of inferred monthly salary")
                elif amount > monthly_salary * 0.9:
                    scores[transaction_id] += 0.3
                    reasons[transaction_id].append("transaction is large relative to inferred monthly salary")
            if user and user.vulnerability_score > 0.2 and row["_transaction_type"].lower() in {"withdrawal", "e-commerce"}:
                scores[transaction_id] += user.vulnerability_score * 0.15
                reasons[transaction_id].append("sender profile suggests elevated susceptibility to social-engineering attacks")

            if pd.notna(timestamp):
                window = hour_windows.get(str(row["_sender_iban"]))
                hour = float(timestamp.hour)
                if window and (hour < window[0] or hour > window[1]):
                    scores[transaction_id] += 0.15
                    reasons[transaction_id].append("transaction time is outside the sender's usual hours")
                elif hour < 5 or hour > 23:
                    scores[transaction_id] += 0.12
                    reasons[transaction_id].append("transaction occurred at an unusual night-time hour")

            sender_country = row["_sender_iban"][:2]
            recipient_country = row["_recipient_iban"][:2]
            if sender_country and recipient_country and sender_country != recipient_country and pd.notna(amount):
                sender_history = tx.loc[tx["_sender_iban"] == row["_sender_iban"], "_recipient_iban"].dropna().astype(str)
                recipient_countries = sender_history.str[:2]
                if int((recipient_countries == recipient_country).sum()) <= 1 and amount > tx["_amount"].quantile(0.75):
                    scores[transaction_id] += 0.2
                    reasons[transaction_id].append("cross-border recipient is rare for this sender")

        result["score"] = result["transaction_id"].map(scores).fillna(0.0).clip(0.0, 1.0)
        result["reasons"] = result["transaction_id"].map(lambda transaction_id: reasons.get(transaction_id, []))
        return AgentResult(self.name, result)


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lng = radians(lng2 - lng1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lng / 2) ** 2
    return 2 * radius * asin(sqrt(a))


def extract_city(location_value: str) -> str | None:
    cleaned = (location_value or "").strip()
    if not cleaned:
        return None
    if " - " in cleaned:
        return cleaned.split(" - ")[0].strip()
    if "online" in cleaned.lower():
        return None
    return cleaned


class LocationRiskAgent(BaseAgent):
    name = "location_risk"

    def resolve_user(self, row: pd.Series, context: DatasetContext) -> UserProfile | None:
        sender_iban = row["_sender_iban"]
        sender_id = row["_sender_id"]
        if sender_iban and sender_iban in context.user_by_iban:
            return context.user_by_iban[sender_iban]
        if sender_id and sender_id in context.user_by_biotag:
            return context.user_by_biotag[sender_id]
        return None

    def analyze(self, context: DatasetContext) -> AgentResult:
        result = self.empty_result(context)
        if context.locations.empty:
            return AgentResult(self.name, result)

        tx = context.transactions
        tx_id_col = context.tx_cols["transaction_id"]
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, list[str]] = defaultdict(list)
        city_centers = (
            context.locations.dropna(subset=["city", "lat", "lng"])
            .groupby("city", dropna=True)
            .agg(lat=("lat", "median"), lng=("lng", "median"))
            .to_dict(orient="index")
        )
        traces = {biotag: df.sort_values("timestamp") for biotag, df in context.locations.groupby("biotag", dropna=True)}

        for _, row in tx.iterrows():
            transaction_id = row[tx_id_col]
            user = self.resolve_user(row, context)
            tx_city = extract_city(row["_location"])
            timestamp = row["_timestamp"]
            tx_type = row["_transaction_type"].lower()
            payment_method = row["_payment_method"].lower()

            if not user or not user.biotag or user.biotag not in traces:
                continue
            if tx_type == "e-commerce" or "paypal" in payment_method or not tx_city or tx_city not in city_centers:
                continue

            tx_lat = float(city_centers[tx_city]["lat"])
            tx_lng = float(city_centers[tx_city]["lng"])
            if user.residence_city and tx_city.lower() != user.residence_city.lower():
                scores[transaction_id] += 0.2
                reasons[transaction_id].append("payment city differs from the user's home city")
            if user.residence_lat is not None and user.residence_lng is not None:
                home_distance = haversine_km(user.residence_lat, user.residence_lng, tx_lat, tx_lng)
                if home_distance > 120:
                    scores[transaction_id] += 0.25
                    reasons[transaction_id].append("payment happens far from the user's residence")
            if pd.notna(timestamp):
                recent_trace = traces[user.biotag]
                recent_trace = recent_trace.loc[
                    (recent_trace["timestamp"] <= timestamp) & (recent_trace["timestamp"] >= timestamp - pd.Timedelta(hours=12))
                ]
                if not recent_trace.empty:
                    latest = recent_trace.iloc[-1]
                    trace_distance = haversine_km(float(latest["lat"]), float(latest["lng"]), tx_lat, tx_lng)
                    if trace_distance > 150:
                        scores[transaction_id] += 0.45
                        reasons[transaction_id].append("payment location conflicts with recent device trace")

        result["score"] = result["transaction_id"].map(scores).fillna(0.0).clip(0.0, 1.0)
        result["reasons"] = result["transaction_id"].map(lambda transaction_id: reasons.get(transaction_id, []))
        return AgentResult(self.name, result)


def suspicious_domain_score(domain: str) -> float:
    score = 0.0
    if not domain:
        return score
    if "bit.ly" in domain:
        score += 0.35
    if re.search(r"[01345]", domain):
        score += 0.25
    if any(token in domain for token in ("verify", "secure", "confirm", "billing", "renew")):
        score += 0.15
    if any(token in domain for token in ("paypal", "amazon", "uber", "netflix", "bank")) and not any(
        trusted in domain for trusted in ("paypal.com", "amazon.com", "edf.fr", "deutschebank.com", "barclays.co.uk")
    ):
        score += 0.25
    if "social" in domain or "pension" in domain or "benefit" in domain:
        score += 0.15
    return min(score, 0.8)


def is_trusted_domain(domain: str) -> bool:
    domain = domain.lower().strip()
    if not domain:
        return False
    return any(domain == suffix or domain.endswith(f".{suffix}") for suffix in TRUSTED_DOMAIN_SUFFIXES)


def communication_score(text: str) -> tuple[float, list[str]]:
    normalized = " ".join(text.split())
    if TRAINING_SIMULATION_PATTERN.search(normalized):
        return 0.0, []
    urls = URL_PATTERN.findall(normalized)
    bare_urls = [match.group(0) for match in BARE_URL_PATTERN.finditer(normalized) if not match.group(0).lower().startswith("http")]
    score = 0.0
    benign_credit = 0.0
    reasons: list[str] = []

    if URGENCY_PATTERN.search(normalized):
        score += 0.2
        reasons.append("contains urgency or account-verification language")
    if SENSITIVE_BRAND_PATTERN.search(normalized):
        score += 0.12
        reasons.append("targets a financial, benefits, or account-security theme")
    if PAYMENT_ACTION_PATTERN.search(normalized):
        score += 0.22
        reasons.append("requests payment, credential, or account-confirmation action")
    if BENIGN_COMMUNICATION_PATTERN.search(normalized):
        benign_credit += 0.22
    if SAFETY_ASSURANCE_PATTERN.search(normalized):
        benign_credit += 0.28
    if bare_urls:
        score += 0.15
        reasons.append("contains bare domains or short links often used in phishing")
    trusted_link_count = 0
    for url in urls:
        domain = urlparse(url).netloc.lower()
        domain_score = suspicious_domain_score(domain)
        if domain_score > 0:
            score += domain_score
            reasons.append(f"links to suspicious domain {domain}")
        elif is_trusted_domain(domain):
            trusted_link_count += 1
            benign_credit += 0.08
        if url.lower().startswith("http://"):
            score += 0.15
            reasons.append("uses insecure http link")
    for url in bare_urls:
        domain = url.split("/")[0].lower()
        domain_score = suspicious_domain_score(domain)
        score += domain_score
        if domain_score <= 0 and is_trusted_domain(domain):
            trusted_link_count += 1
            benign_credit += 0.08

    if urls and trusted_link_count == len(urls) and not bare_urls:
        benign_credit += 0.10
    if score < 0.22:
        return 0.0, []

    final_score = max(0.0, min(score - benign_credit, 1.0))
    if final_score < 0.18:
        return 0.0, []
    return final_score, reasons


class CommunicationRiskAgent(BaseAgent):
    name = "communication_risk"

    def resolve_user(self, row: pd.Series, context: DatasetContext) -> UserProfile | None:
        sender_iban = row["_sender_iban"]
        sender_id = row["_sender_id"]
        if sender_iban and sender_iban in context.user_by_iban:
            return context.user_by_iban[sender_iban]
        if sender_id and sender_id in context.user_by_biotag:
            return context.user_by_biotag[sender_id]
        return None

    def assign_user(self, text: str, context: DatasetContext) -> UserProfile | None:
        normalized_text = normalize_name_token(text)
        for user in context.users:
            if normalize_name_token(user.full_name) and normalize_name_token(user.full_name) in normalized_text:
                return user
            if normalize_name_token(user.first_name) and normalize_name_token(user.first_name) in normalized_text:
                if not user.residence_city or normalize_name_token(user.residence_city) in normalized_text:
                    return user
        return None

    def build_events(self, context: DatasetContext) -> pd.DataFrame:
        events: list[dict[str, object]] = []

        for item in context.sms_messages:
            text = str(item.get("sms", ""))
            score, reasons = communication_score(text)
            if score <= 0:
                continue
            match = SMS_DATE_PATTERN.search(text)
            timestamp = parse_timestamp(match.group(1)) if match else pd.NaT
            user = self.assign_user(text, context)
            events.append({"user_key": user.user_key if user else None, "timestamp": timestamp, "score": score, "channel": "sms", "reasons": reasons})

        for item in context.mail_messages:
            text = str(item.get("mail", ""))
            score, reasons = communication_score(text)
            if score <= 0:
                continue
            match = MAIL_DATE_PATTERN.search(text)
            timestamp = parse_timestamp(match.group(1)) if match else pd.NaT
            user = self.assign_user(text, context)
            events.append({"user_key": user.user_key if user else None, "timestamp": timestamp, "score": score, "channel": "mail", "reasons": reasons})

        return pd.DataFrame(events)

    def analyze(self, context: DatasetContext) -> AgentResult:
        result = self.empty_result(context)
        events = self.build_events(context)
        if events.empty:
            return AgentResult(self.name, result)

        tx_id_col = context.tx_cols["transaction_id"]
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, list[str]] = defaultdict(list)

        for _, row in context.transactions.iterrows():
            transaction_id = row[tx_id_col]
            user = self.resolve_user(row, context)
            timestamp = row["_timestamp"]
            if not user or pd.isna(timestamp):
                continue

            dataset_id = canonical_dataset_id(context.dataset_name)
            window_days = 12 if dataset_id in {"l4", "l5"} else 21 if dataset_id == "l3" else 30
            relevant = events.loc[
                (events["user_key"] == user.user_key)
                & (events["timestamp"] <= timestamp)
                & (events["timestamp"] >= timestamp - pd.Timedelta(days=window_days))
            ].copy()
            if relevant.empty:
                continue

            relevant["days_before"] = (timestamp - relevant["timestamp"]).dt.total_seconds() / 86400.0
            relevant["decayed"] = relevant["score"] * (
                1.0 - relevant["days_before"].clip(lower=0, upper=window_days) / max(window_days + 2, 1)
            )
            strongest = relevant.sort_values("decayed", ascending=False).iloc[0]
            if float(strongest["decayed"]) < 0.24:
                continue
            tx_type = row["_transaction_type"].lower()
            type_multiplier = 1.0
            if tx_type in {"withdrawal", "e-commerce"}:
                type_multiplier = 1.25
            elif tx_type in {"transfer", "direct debit"}:
                type_multiplier = 1.10
            user_multiplier = 1.0 + (user.vulnerability_score if user else 0.0)
            scores[transaction_id] += min(float(strongest["decayed"]) * type_multiplier * user_multiplier, 0.75)
            if int((relevant["score"] >= 0.45).sum()) >= 2:
                scores[transaction_id] += 0.15
                reasons[transaction_id].append("multiple suspicious communications preceded this transaction")
            if tx_type in {"withdrawal", "e-commerce"} and int((relevant["score"] >= 0.55).sum()) >= 1:
                scores[transaction_id] += 0.12
                reasons[transaction_id].append("high-risk phishing communication preceded a cash-out or card-not-present transaction")
            reasons[transaction_id].append(
                f"recent {strongest['channel']} showed phishing signals: {', '.join(strongest['reasons'][:2])}"
            )

        result["score"] = result["transaction_id"].map(scores).fillna(0.0).clip(0.0, 1.0)
        result["reasons"] = result["transaction_id"].map(lambda transaction_id: reasons.get(transaction_id, []))
        return AgentResult(self.name, result)


class AudioRiskAgent(BaseAgent):
    name = "audio_risk"

    def resolve_user(self, row: pd.Series, context: DatasetContext) -> UserProfile | None:
        sender_iban = row["_sender_iban"]
        sender_id = row["_sender_id"]
        if sender_iban and sender_iban in context.user_by_iban:
            return context.user_by_iban[sender_iban]
        if sender_id and sender_id in context.user_by_biotag:
            return context.user_by_biotag[sender_id]
        return None

    def analyze(self, context: DatasetContext) -> AgentResult:
        result = self.empty_result(context)
        if not context.audio_events:
            return AgentResult(self.name, result)

        events = pd.DataFrame(context.audio_events)
        events = events.loc[events["user_key"].notna()].copy()
        if events.empty:
            return AgentResult(self.name, result)

        tx = context.transactions.sort_values("_timestamp", kind="stable")
        tx_id_col = context.tx_cols["transaction_id"]
        amount_baselines = tx.groupby("_sender_iban")["_amount"].median().to_dict()
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, list[str]] = defaultdict(list)

        for _, row in tx.iterrows():
            transaction_id = row[tx_id_col]
            timestamp = row["_timestamp"]
            user = self.resolve_user(row, context)
            if user is None or pd.isna(timestamp):
                continue

            relevant = events.loc[
                (events["user_key"] == user.user_key)
                & (events["timestamp"] <= timestamp)
                & (events["timestamp"] >= timestamp - pd.Timedelta(hours=72))
            ].copy()
            if relevant.empty:
                continue

            relevant["hours_before"] = (timestamp - relevant["timestamp"]).dt.total_seconds() / 3600.0
            strongest = relevant.sort_values("hours_before", ascending=True).iloc[0]
            hours_before = float(strongest["hours_before"])
            score = 0.0

            if hours_before <= 6:
                score += 0.18
                reasons[transaction_id].append("transaction followed a recent call within 6 hours")
            elif hours_before <= 24:
                score += 0.12
                reasons[transaction_id].append("transaction followed a recent call within 24 hours")
            else:
                score += 0.06
                reasons[transaction_id].append("transaction followed a recent call within 72 hours")

            tx_type = row["_transaction_type"].lower()
            if tx_type in {"transfer", "withdrawal", "direct debit"}:
                score += 0.10
                reasons[transaction_id].append("voice contact preceded a cash-moving transaction")
            elif tx_type == "e-commerce":
                score += 0.06

            amount = row["_amount"]
            sender_median = amount_baselines.get(row["_sender_iban"])
            if pd.notna(amount) and sender_median and amount > max(sender_median * 1.8, 500):
                score += 0.10
                reasons[transaction_id].append("post-call transaction is large relative to sender baseline")

            if pd.notna(timestamp) and (timestamp.hour < 6 or timestamp.hour > 22):
                score += 0.05

            scores[transaction_id] += min(score, 0.45)

        result["score"] = result["transaction_id"].map(scores).fillna(0.0).clip(0.0, 1.0)
        result["reasons"] = result["transaction_id"].map(lambda transaction_id: reasons.get(transaction_id, []))
        return AgentResult(self.name, result)


class DecisionAgent:
    weights = {
        "transaction_pattern": 0.32,
        "user_behavior": 0.23,
        "location_risk": 0.17,
        "communication_risk": 0.28,
        "audio_risk": 0.0,
    }

    def decide(self, context: DatasetContext, agent_results: list[AgentResult]) -> pd.DataFrame:
        tx = context.transactions.copy()
        tx_id_col = context.tx_cols["transaction_id"]
        merged = pd.DataFrame(
            {
                "transaction_id": tx[tx_id_col],
                "transaction_type": tx["_transaction_type"],
                "amount": tx["_amount"],
                "sender_iban": tx["_sender_iban"],
                "recipient_iban": tx["_recipient_iban"],
                "description": tx["_description"],
            }
        )
        merged["reasons"] = [[] for _ in range(len(merged))]

        for agent_result in agent_results:
            renamed = agent_result.frame.rename(columns={"score": f"{agent_result.name}_score", "reasons": f"{agent_result.name}_reasons"})
            merged = merged.merge(renamed, on="transaction_id", how="left")
            merged[f"{agent_result.name}_score"] = merged[f"{agent_result.name}_score"].fillna(0.0)
            merged[f"{agent_result.name}_reasons"] = merged[f"{agent_result.name}_reasons"].apply(lambda value: value if isinstance(value, list) else [])

        weights = self.weights.copy()
        dataset_id = canonical_dataset_id(context.dataset_name)
        if dataset_id in {"l4", "l5"}:
            weights = {
                "transaction_pattern": 0.34,
                "user_behavior": 0.21,
                "location_risk": 0.20,
                "communication_risk": 0.13,
                "audio_risk": 0.12,
            }

        merged["final_score"] = 0.0
        for agent_result in agent_results:
            merged["final_score"] += merged[f"{agent_result.name}_score"] * weights.get(agent_result.name, 0.0)
            merged["reasons"] = merged.apply(
                lambda row, agent_name=agent_result.name: row["reasons"] + row[f"{agent_name}_reasons"],
                axis=1,
            )

        amount_pct = merged["amount"].rank(pct=True, method="average").fillna(0.0)
        merged["economic_priority"] = amount_pct
        if dataset_id != "l3":
            merged.loc[merged["transaction_type"].str.lower() == "transfer", "economic_priority"] *= 1.35
            merged.loc[merged["transaction_type"].str.lower() == "direct debit", "economic_priority"] *= 1.15
            merged.loc[merged["transaction_type"].str.lower() == "withdrawal", "economic_priority"] *= 0.85
            merged.loc[merged["transaction_type"].str.lower() == "e-commerce", "economic_priority"] *= 0.8
            merged["economic_priority"] = merged["economic_priority"].clip(0.0, 1.0)
            merged["final_score"] += merged["economic_priority"] * 0.12

        if dataset_id in {"l4", "l5"}:
            tx_type_lower = merged["transaction_type"].str.lower()
            weak_small_cashout = (
                tx_type_lower.isin(["withdrawal", "direct debit", "e-commerce", "in-person payment"])
                & (merged["amount"].fillna(0.0) < 275)
                & (merged["transaction_pattern_score"] < 0.20)
                & (merged["location_risk_score"] < 0.20)
                & (merged["audio_risk_score"] < 0.10)
            )
            merged.loc[weak_small_cashout, "final_score"] -= 0.035

            comm_only_transfer = (
                tx_type_lower.eq("transfer")
                & (merged["amount"].fillna(0.0) < 350)
                & (merged["transaction_pattern_score"] < 0.20)
                & (merged["location_risk_score"] < 0.05)
                & (merged["audio_risk_score"] < 0.10)
                & (merged["communication_risk_score"] > 0.75)
            )
            merged.loc[comm_only_transfer, "final_score"] -= 0.025

        if dataset_id == "l5":
            pair_counts = (
                tx.groupby(["_sender_iban", "_recipient_iban"], dropna=False)
                .size()
                .rename("pair_count")
                .reset_index()
                .rename(columns={"_sender_iban": "sender_iban", "_recipient_iban": "recipient_iban"})
            )
            merged = merged.merge(pair_counts, on=["sender_iban", "recipient_iban"], how="left")
            merged["pair_count"] = merged["pair_count"].fillna(0)
            description_text = merged["description"].fillna("").astype(str)
            amount_high = merged["amount"].fillna(0.0).rank(pct=True, method="average")

            recurring_obligation = (
                tx_type_lower.isin(["transfer", "direct debit"])
                & (merged["pair_count"] >= 2)
                & description_text.str.contains(LEGITIMATE_OBLIGATION_PATTERN, regex=True)
                & (merged["transaction_pattern_score"] < 0.45)
                & (merged["location_risk_score"] < 0.15)
                & (merged["audio_risk_score"] < 0.15)
            )
            merged.loc[recurring_obligation, "final_score"] -= 0.055

            strongly_recurring_obligation = (
                tx_type_lower.isin(["transfer", "direct debit"])
                & (merged["pair_count"] >= 3)
                & description_text.str.contains(LEGITIMATE_OBLIGATION_PATTERN, regex=True)
                & (merged["transaction_pattern_score"] < 0.60)
                & (merged["location_risk_score"] < 0.15)
                & (merged["audio_risk_score"] < 0.15)
            )
            merged.loc[strongly_recurring_obligation, "final_score"] -= 0.035

            recurring_comm_only = (
                tx_type_lower.isin(["transfer", "direct debit", "withdrawal", "e-commerce"])
                & (merged["pair_count"] >= 2)
                & (merged["communication_risk_score"] > 0.75)
                & (merged["transaction_pattern_score"] < 0.20)
                & (merged["location_risk_score"] < 0.05)
                & (merged["audio_risk_score"] < 0.10)
            )
            merged.loc[recurring_comm_only, "final_score"] -= 0.035

            likely_internal_or_household = (
                tx_type_lower.eq("transfer")
                & description_text.str.contains(r"\b(?:savings|emergency fund|donation|rent payment|family transfer|gift to relative)\b", case=False, regex=True)
                & (merged["pair_count"] >= 1)
                & (merged["transaction_pattern_score"] < 0.65)
                & (merged["location_risk_score"] < 0.10)
                & (merged["audio_risk_score"] < 0.10)
            )
            merged.loc[likely_internal_or_household, "final_score"] -= 0.045

            high_value_supported = (
                tx_type_lower.isin(["transfer", "withdrawal", "e-commerce"])
                & (amount_high >= 0.92)
                & (
                    (merged["transaction_pattern_score"] >= 0.55)
                    | (merged["location_risk_score"] >= 0.30)
                    | (merged["audio_risk_score"] >= 0.12)
                    | (merged["communication_risk_score"] >= 0.60)
                )
                & ~description_text.str.contains(LEGITIMATE_OBLIGATION_PATTERN, regex=True)
            )
            merged.loc[high_value_supported, "final_score"] += 0.025

            high_value_cashout = (
                tx_type_lower.isin(["withdrawal", "e-commerce"])
                & (amount_high >= 0.88)
                & (
                    (merged["communication_risk_score"] >= 0.70)
                    | (merged["location_risk_score"] >= 0.30)
                    | (merged["transaction_pattern_score"] >= 0.45)
                )
            )
            merged.loc[high_value_cashout, "final_score"] += 0.018

            corroborated_large_transfer = (
                tx_type_lower.eq("transfer")
                & (amount_high >= 0.90)
                & (merged["transaction_pattern_score"] >= 0.45)
                & (
                    (merged["communication_risk_score"] >= 0.55)
                    | (merged["location_risk_score"] >= 0.20)
                    | (merged["audio_risk_score"] >= 0.12)
                )
                & ~description_text.str.contains(LEGITIMATE_OBLIGATION_PATTERN, regex=True)
            )
            merged.loc[corroborated_large_transfer, "final_score"] += 0.012

        merged["final_score"] = merged["final_score"].clip(0.0, 1.0)
        merged = merged.sort_values("final_score", ascending=False, kind="stable").reset_index(drop=True)

        total = len(merged)
        if dataset_id == "l3":
            min_flags = max(3, ceil(total * 0.06))
            max_flags = max(min_flags, ceil(total * 0.30))
            threshold = max(0.22, float(merged["final_score"].quantile(0.80)))
        elif dataset_id in {"l4", "l5"}:
            min_flags = max(3, ceil(total * 0.05))
            max_flags = max(min_flags, ceil(total * 0.16))
            threshold = max(0.24, float(merged["final_score"].quantile(0.88)))
        else:
            min_flags = max(3, ceil(total * 0.06))
            max_flags = max(min_flags, ceil(total * 0.22))
            threshold = max(0.26, float(merged["final_score"].quantile(0.84)))
        flagged = merged.loc[merged["final_score"] >= threshold].copy()
        if len(flagged) < min_flags:
            flagged = merged.head(min_flags).copy()
        elif len(flagged) > max_flags:
            flagged = merged.head(max_flags).copy()

        if dataset_id != "l3":
            transfer_pool = merged.loc[
                merged["transaction_type"].str.lower().isin(["transfer", "direct debit"]) & (merged["economic_priority"] >= 0.70)
            ].copy()
            transfer_pool = transfer_pool.sort_values(["final_score", "economic_priority"], ascending=False)
            reserve_count = max(4, ceil(len(flagged) * 0.25))
            reserve = transfer_pool.head(reserve_count)
            if not reserve.empty:
                flagged = (
                    pd.concat([flagged, reserve], ignore_index=True)
                    .sort_values(["final_score", "economic_priority"], ascending=False, kind="stable")
                    .drop_duplicates(subset=["transaction_id"])
                    .head(max_flags)
                    .copy()
                )

        merged["is_flagged"] = merged["transaction_id"].isin(flagged["transaction_id"])
        return merged


def write_output(path: Path, decision: pd.DataFrame) -> None:
    flagged_ids = decision.loc[decision["is_flagged"], "transaction_id"].astype(str).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(flagged_ids) + ("\n" if flagged_ids else ""), encoding="utf-8")


def run_pipeline(dataset_path: Path, dataset_name: str, output_path: Path) -> tuple[str, pd.DataFrame]:
    load_project_env()
    langfuse_client = build_langfuse()
    session_id = generate_session_id()

    span_context = nullcontext(None)
    if langfuse_client:
        try:
            span_context = langfuse_client.start_as_current_span(
                name="fraud-detection-pipeline",
                input={"dataset_name": dataset_name, "dataset_path": str(dataset_path), "output_path": str(output_path)},
                metadata={"pipeline": "l1-agentic"},
            )
        except Exception as exc:
            print(f"Tracing disabled: could not start Langfuse span ({exc})")
            langfuse_client = None

    with span_context as root_span:
        if root_span and langfuse_client:
            try:
                langfuse_client.update_current_trace(
                    name=f"fraud-detection-{dataset_name}",
                    session_id=session_id,
                    input={"dataset": dataset_name, "dataset_path": str(dataset_path)},
                    metadata={"output_path": str(output_path)},
                    tags=[dataset_name, "fraud-detection"],
                )
            except Exception as exc:
                print(f"Tracing warning: could not update Langfuse trace ({exc})")

        context = load_context(dataset_path, dataset_name)
        agents = [
            TransactionPatternAgent(),
            UserBehaviorAgent(),
            LocationRiskAgent(),
            CommunicationRiskAgent(),
            AudioRiskAgent(),
        ]
        results: list[AgentResult] = []
        for agent in agents:
            if root_span:
                with root_span.start_as_current_observation(name=f"agent:{agent.name}", as_type="span") as agent_span:
                    agent_result = agent.analyze(context)
                    agent_span.update(
                        output={
                            "max_score": float(agent_result.frame["score"].max()),
                            "mean_score": float(agent_result.frame["score"].mean()),
                            "non_zero": int((agent_result.frame["score"] > 0).sum()),
                        }
                    )
            else:
                agent_result = agent.analyze(context)
            results.append(agent_result)

        decision = DecisionAgent().decide(context, results)
        write_output(output_path, decision)
        if root_span:
            root_span.update(output={"flagged_count": int(decision["is_flagged"].sum()), "session_id": session_id})
        if langfuse_client:
            try:
                langfuse_client.flush()
            except Exception as exc:
                print(f"Tracing warning: could not flush Langfuse events ({exc})")
        return session_id, decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent-based fraud detection pipeline for Reply challenge dataset l1")
    parser.add_argument("--dataset", default="dataset/the-truman-show", help="Dataset directory path")
    parser.add_argument("--dataset-name", default="l1", help="Dataset label")
    parser.add_argument("--output", default="outputs/the-truman-show.txt", help="Output text file")
    return parser


def run_cli(default_dataset: str, default_dataset_name: str, default_output: str) -> None:
    parser = argparse.ArgumentParser(description=f"Agent-based fraud detection pipeline for Reply challenge dataset {default_dataset_name}")
    parser.add_argument("--dataset", default=default_dataset, help="Dataset directory path")
    parser.add_argument("--dataset-name", default=default_dataset_name, help="Dataset label")
    parser.add_argument("--output", default=default_output, help="Output text file")
    args = parser.parse_args()
    output_path = Path(args.output)
    session_id, decision = run_pipeline(Path(args.dataset), args.dataset_name, output_path)
    flagged = decision.loc[decision["is_flagged"], ["transaction_id", "final_score"]]
    print(f"\nFlagged transactions: {len(flagged)}")
    print(f"Team prefix: {resolve_team_name()}")
    print(f"Session ID: {session_id}")
    print(f"Output file: {output_path.resolve()}")
    print("Top suspicious transactions:")
    for row in flagged.head(10).to_dict(orient="records"):
        print(f"  {row['transaction_id']} score={row['final_score']:.3f}")


def main() -> None:
    run_cli("../dataset/l1", "l1", "output_l1.txt")


if __name__ == "__main__":
    main()
