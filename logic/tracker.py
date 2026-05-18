"""자산 트래커 로직 + Firebase 클라우드 연동"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import requests


# ============================================================
# 색상 (토스 톤다운 팔레트)
# ============================================================
COLOR_CASH = "#7CB342"       # 현금성 - 연두
COLOR_DOLLAR = "#2E7D32"     # 달러 - 진초록
COLOR_LEVERAGE = "#8D2A1F"   # 레버리지 - 적갈
COLOR_NASDAQ = "#E53935"     # 나스닥 - 빨강
COLOR_SPY = "#FB8C00"        # SPY - 주황
COLOR_DIVIDEND = "#FDD835"   # 배당주 - 노랑
COLOR_OTHER = [
    "#3182F6", "#7E57C2", "#26A69A", "#EC407A",
    "#5C6BC0", "#42A5F5", "#5E35B1", "#00897B",
]


# ============================================================
# 텍스트 정제
# ============================================================
def sanitize_key(s: str) -> str:
    if not s:
        return ""
    for tok in ("LBRB", "LB", "RB", "_HASH_", "_DOT_"):
        s = s.replace(tok, "")
    return re.sub(r"[^a-zA-Z0-9가-힣]", "", s)


def clean_tag_name(s: str) -> str:
    if not s:
        return ""
    for tok in ("LBRB", "LB", "RB", "_HASH_", "_DOT_", "_DOL_", "_SL_"):
        s = s.replace(tok, "")
    return re.sub(r"[^a-zA-Z0-9가-힣]", "", s)


# ============================================================
# 자산 분류
# ============================================================
def get_asset_type(tag: str) -> str:
    lo = tag.lower()
    if lo in ("달러", "usd", "dollar"):
        return "dollar"
    if (lo == "현금" or "예금" in lo or "적금" in lo or "예적금" in lo
            or "채권" in lo or lo == "rp" or "저축" in lo or lo in ("cma", "mmf")
            or "파킹" in lo or "입출금" in lo):
        return "cash"
    if lo in ("tqqq", "qld", "upro", "soxl", "tecl", "fngu", "bulz", "sso") \
            or "레버리지" in lo or "3x" in lo or "2x" in lo:
        return "leverage"
    if lo in ("qqq", "qqqm") or "나스닥" in lo:
        return "nasdaq"
    if lo in ("spy", "voo", "ivv", "splg") or "s&p" in lo or "sp500" in lo or "snp" in lo:
        return "spy"
    if lo in ("msft", "schd", "vym", "dgro", "aapl", "ko", "jnj", "pg",
              "vti", "vtv", "vug", "dia") or "배당" in lo or "dividend" in lo:
        return "dividend"
    return "other"


def get_super_group(t: str) -> str:
    if t in ("spy", "dividend"):
        return "spy_div"
    if t in ("cash", "dollar"):
        return "cash_dol"
    if t in ("leverage", "nasdaq"):
        return "lev_nas"
    return "other_grp"


def get_color_for_tag(tag: str) -> Optional[str]:
    t = get_asset_type(tag)
    return {
        "cash": COLOR_CASH, "dollar": COLOR_DOLLAR,
        "leverage": COLOR_LEVERAGE, "nasdaq": COLOR_NASDAQ,
        "spy": COLOR_SPY, "dividend": COLOR_DIVIDEND,
    }.get(t)


def assign_colors(tag_list: List[str]) -> Dict[str, str]:
    colors: Dict[str, str] = {}
    other_idx = 0
    for tag in tag_list:
        c = get_color_for_tag(tag)
        if c:
            colors[tag] = c
        else:
            colors[tag] = COLOR_OTHER[other_idx % len(COLOR_OTHER)]
            other_idx += 1
    return colors


# ============================================================
# 슈퍼그룹 정렬 (원본 3단 정렬)
# ============================================================
def sort_tags_by_super_group(entries: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    sg_totals = {"spy_div": 0.0, "cash_dol": 0.0, "lev_nas": 0.0, "other_grp": 0.0}
    type_totals: Dict[str, float] = {}
    type_map: Dict[str, str] = {}

    for tag, val in entries:
        t = get_asset_type(tag)
        type_map[tag] = t
        sg_totals[get_super_group(t)] += val
        type_totals[t] = type_totals.get(t, 0.0) + val

    def key(item):
        tag, val = item
        t = type_map[tag]
        sg = get_super_group(t)
        return (-sg_totals[sg], -type_totals[t], -val)

    return sorted(entries, key=key)


# ============================================================
# 뱅크샐러드 파싱
# ============================================================
def parse_data(raw: str) -> List[Dict]:
    out = []
    for line in raw.strip().splitlines():
        t = line.strip()
        if not t or "상품명" in t:
            continue
        name = ""
        amount = 0
        parts = t.split("\t")
        if len(parts) >= 2:
            name = parts[0].strip()
            try:
                amount = int(re.sub(r"[,원\s]", "", parts[-1]))
            except ValueError:
                amount = 0
        else:
            m = re.match(r"^(.+?)\s{2,}([\d,]+)", t)
            if m:
                name = m.group(1).strip()
                try:
                    amount = int(m.group(2).replace(",", ""))
                except ValueError:
                    amount = 0
        if name and amount > 0:
            out.append({"name": name, "amount": amount})
    return out


def extract_tag(name: str) -> Optional[str]:
    lo = name.lower()
    if "비트코인" in lo or "bitcoin" in lo or "btc" in lo:
        return "비트코인"
    m = re.search(r"#([\w가-힣]+)", name)
    return m.group(1).upper() if m else None


def process_data(items: List[Dict], etc_threshold: int = 200000) -> Dict[str, int]:
    groups: Dict[str, int] = {}
    etc = 0
    for it in items:
        tag = extract_tag(it["name"])
        if not tag or it["amount"] < etc_threshold:
            etc += it["amount"]
        else:
            safe = sanitize_key(tag)
            groups[safe] = groups.get(safe, 0) + it["amount"]
    if etc > 0:
        groups["기타"] = groups.get("기타", 0) + etc
    return groups


def clean_all_data(data: Dict) -> Dict[str, Dict[str, int]]:
    cleaned: Dict[str, Dict[str, int]] = {}
    for mk, sub in data.items():
        clean_mk = re.sub(r"[^0-9\-]", "", mk)
        if not clean_mk:
            continue
        cleaned.setdefault(clean_mk, {})
        if not isinstance(sub, dict):
            continue
        for tk, v in sub.items():
            clean_tk = sanitize_key(tk)
            if not clean_tk:
                continue
            cleaned[clean_mk][clean_tk] = cleaned[clean_mk].get(clean_tk, 0) + int(v)
    return cleaned


def get_sorted_entries(data: Dict[str, int]) -> List[Tuple[str, int]]:
    entries = [(clean_tag_name(k), v) for k, v in data.items()]
    return sort_tags_by_super_group(entries)


def aggregate_for_trend(asset_data: Dict[str, Dict[str, int]],
                        start_key: str, end_key: str) -> Tuple[List[str], List[Tuple[str, List[float]]]]:
    """월별 추이 차트용 데이터: (labels, [(tag, values_per_month), ...])"""
    keys = sorted([k for k in asset_data if start_key <= k <= end_key])
    if not keys:
        return [], []

    tag_totals: Dict[str, float] = {}
    for k in keys:
        for t, v in asset_data[k].items():
            ct = clean_tag_name(t)
            tag_totals[ct] = tag_totals.get(ct, 0) + v

    sorted_entries = sort_tags_by_super_group(list(tag_totals.items()))
    tag_list = [t for t, _ in sorted_entries]

    series: List[Tuple[str, List[float]]] = []
    for tag in tag_list:
        vals = []
        for k in keys:
            s = 0
            for orig, v in asset_data[k].items():
                if clean_tag_name(orig) == tag:
                    s += v
            vals.append(s)
        series.append((tag, vals))

    labels = []
    for k in keys:
        y, m = k.split("-")
        labels.append(f"{y[2:]}.{int(m)}")
    return labels, series


# ============================================================
# 로컬 캐시
# ============================================================
LOCAL_PATH = os.path.expanduser("~/.asset_tracker_data.json")


def load_local() -> Dict:
    if os.path.exists(LOCAL_PATH):
        try:
            with open(LOCAL_PATH, "r", encoding="utf-8") as f:
                return clean_all_data(json.load(f))
        except Exception:
            return {}
    return {}


def save_local(data: Dict) -> None:
    try:
        with open(LOCAL_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ============================================================
# Firebase REST 클라이언트
# ============================================================
@dataclass
class FirebaseClient:
    base_url: str = ""

    def _normalize(self) -> str:
        return self.base_url.rstrip("/")

    def get(self) -> Dict:
        url = self._normalize() + "/assetData.json"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        return clean_all_data(data) if data else {}

    def put(self, data: Dict) -> None:
        url = self._normalize() + "/assetData.json"
        r = requests.put(url, json=data, timeout=15)
        r.raise_for_status()

    def delete(self) -> None:
        url = self._normalize() + "/assetData.json"
        r = requests.delete(url, timeout=15)
        r.raise_for_status()
