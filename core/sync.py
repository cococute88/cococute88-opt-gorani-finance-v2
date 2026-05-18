# core/sync.py
import streamlit as st
import json
import time
from datetime import datetime
from core.firebase import load_data, save_data


# ---------------------------------------------------------------------------
# Firebase 호환을 위한 키 정제
# ---------------------------------------------------------------------------
_FORBIDDEN_KEY_CHARS = (".", "$", "#", "[", "]", "/")

def _safe_key(k) -> str:
    """Firebase가 거부하는 문자 -> '_' 치환"""
    s = str(k) if k is not None else "_NONE_"
    for ch in _FORBIDDEN_KEY_CHARS:
        s = s.replace(ch, "_")
    if not s:
        s = "_EMPTY_KEY_"
    return s

def _safe_uid(uid: str) -> str:
    """이메일 형태의 UID에서 마침표(.)나 골뱅이(@)를 언더바(_)로 치환하여 안전한 경로 생성"""
    if not uid:
        return "unknown_user"
    return str(uid).replace("@", "_").replace(".", "_")

def sanitize_firebase_keys(d):
    """딕셔너리/리스트 트리 전체를 재귀 정제. 빈 dict/list 는 더미값으로 대체."""
    if isinstance(d, dict):
        if not d:
            return {"_EMPTY_DICT_": True}
        new_dict = {}
        for k, v in d.items():
            new_dict[_safe_key(k)] = sanitize_firebase_keys(v)
        return new_dict
    if isinstance(d, list):
        if not d:
            return ["_EMPTY_"]
        return [sanitize_firebase_keys(i) for i in d]
    if isinstance(d, (str, int, float, bool)) or d is None:
        return d
    # 기타 타입 (datetime, date 등)은 문자열로 직렬화
    try:
        return json.loads(json.dumps(d, default=str))
    except Exception:
        return str(d)


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------
def load_all_data(raw_uid):
    """앱 접속 시 모든 데이터를 한 번에 로드합니다."""
    if "data_loaded" not in st.session_state:
        # 이메일 UID의 특수문자를 언더바로 치환
        uid = _safe_uid(raw_uid)
        
        with st.spinner("☁️ 클라우드 데이터 동기화 중..."):
            st.session_state["asset_data"] = load_data(uid, "tracker") or {}
            st.session_state["tracker_cfg"] = load_data(uid, "tracker_config") or {}
            st.session_state["sim_cfg"] = load_data(uid, "sim_config") or {}

            # 배당 캘린더 데이터 로드 (방어적 디코딩)
            raw_cal = load_data(uid, "dividend_calendar") or {}
            st.session_state["div_calendar"] = _decode_calendar(raw_cal)
            
            # 새로고침 시 마지막 동기화 시간 UI 변수 복원
            if "_last_sync" in raw_cal:
                st.session_state["div_calendar_last_sync"] = raw_cal["_last_sync"]

            st.session_state["data_loaded"] = True


def _decode_calendar(raw: dict) -> dict:
    """Firebase 에 저장된 더미 토큰(_EMPTY_ / _EMPTY_DICT_ / _EMPTY_PF_)들을 원래 형태로 복원."""
    if not isinstance(raw, dict):
        return {}

    out = {}
    for k, v in raw.items():
        if k.startswith("_EMPTY_") or k == "_last_sync":
            continue
        out[k] = v

    # portfolios 복원
    pfs = out.get("portfolios", {}) or {}
    clean_pfs = {}
    if isinstance(pfs, dict):
        for pf_name, tkrs in pfs.items():
            if pf_name == "_EMPTY_PF_":
                continue
            if isinstance(tkrs, list):
                clean_pfs[pf_name] = [t for t in tkrs if t and t != "_EMPTY_"]
            elif isinstance(tkrs, dict):
                # Firebase 가 list 를 dict 로 반환하는 경우 (인덱스 키)
                vals = []
                try:
                    for _, t in sorted(tkrs.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
                        if t and t != "_EMPTY_":
                            vals.append(t)
                except Exception:
                    vals = [t for t in tkrs.values() if t and t != "_EMPTY_"]
                clean_pfs[pf_name] = vals
            else:
                clean_pfs[pf_name] = []
    out["portfolios"] = clean_pfs

    for sub in ("memos", "marks", "custom_ce"):
        v = out.get(sub, {})
        if not isinstance(v, dict):
            v = {}
        # _EMPTY_DICT_ 토큰 제거
        v = {kk: vv for kk, vv in v.items() if kk != "_EMPTY_DICT_"}
        out[sub] = v

    return out


# ---------------------------------------------------------------------------
# 트래커 / 설정 저장 (기존 로직 유지)
# ---------------------------------------------------------------------------
def auto_save_tracker_data():
    """트래커의 월별 자산 목록 저장"""
    if "user" in st.session_state and "asset_data" in st.session_state:
        raw_uid = st.session_state["user"]["uid"]
        uid = _safe_uid(raw_uid)
        save_data(uid, "tracker", st.session_state["asset_data"])


def auto_save_config(page_name, config_dict):
    """설정값(config)을 클라우드에 즉시 저장"""
    if "user" not in st.session_state:
        return
    raw_uid = st.session_state["user"]["uid"]
    uid = _safe_uid(raw_uid)

    if page_name == "sim":
        path = "sim_config"
    else:
        path = "tracker_config"
    save_data(uid, path, config_dict)


# ---------------------------------------------------------------------------
# 배당 캘린더 저장 (트래커와 동일한 save_data 통로 사용)
# ---------------------------------------------------------------------------
def _build_calendar_payload(raw_data: dict) -> dict:
    """st.session_state['div_calendar'] -> Firebase 에 안전하게 쓸 dict 로 변환."""
    safe = json.loads(json.dumps(raw_data or {}, default=str))
    safe = sanitize_firebase_keys(safe)
    if not isinstance(safe, dict):
        safe = {}

    # 빈 노드 삭제 방지: 모든 4 종 sub-tree 가 항상 존재하도록 보장
    pfs = safe.get("portfolios")
    if not isinstance(pfs, dict) or not pfs:
        safe["portfolios"] = {"_EMPTY_PF_": ["_EMPTY_"]}
    else:
        for pf_name, tkrs in list(pfs.items()):
            if not isinstance(tkrs, list) or len(tkrs) == 0:
                pfs[pf_name] = ["_EMPTY_"]

    for sub in ("memos", "marks", "custom_ce"):
        v = safe.get(sub)
        if not isinstance(v, dict) or not v:
            safe[sub] = {"_EMPTY_DICT_": True}

    safe["_last_sync"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return safe


def auto_save_calendar_data(silent: bool = False) -> bool:
    """배당 캘린더 전용 저장 함수.
    - 트래커가 사용하는 save_data() 통로를 그대로 재사용 (인증/보안 규칙 동일)
    - 어떤 시나리오에서도 노드가 사라지지 않도록 4개 sub-tree 를 항상 dummy 로 채움
    - silent=False 일 경우 실패 사유를 사용자에게 표시
    """
    if "user" not in st.session_state:
        if not silent:
            st.error("❌ 로그인 정보가 없어 캘린더를 저장할 수 없습니다.")
        return False

    raw_uid = st.session_state["user"].get("uid")
    if not raw_uid:
        if not silent:
            st.error("❌ UID 가 없어 캘린더를 저장할 수 없습니다.")
        return False

    # 안전한 UID 생성 (마침표 등 제거)
    uid = _safe_uid(raw_uid)

    if "div_calendar" not in st.session_state:
        st.session_state["div_calendar"] = {}

    raw_data = st.session_state["div_calendar"]

    try:
        payload = _build_calendar_payload(raw_data)
    except Exception as e:
        if not silent:
            st.error(f"❌ 캘린더 직렬화 실패: {e}")
        return False

    # 1차: save_data 공식 통로
    last_err = None
    for attempt in range(3):
        try:
            save_data(uid, "dividend_calendar", payload)
            st.session_state["div_calendar_last_sync"] = payload["_last_sync"]
            st.session_state["div_calendar_last_error"] = None
            return True
        except Exception as e:
            last_err = e
            time.sleep(0.4 * (attempt + 1))

    st.session_state["div_calendar_last_error"] = str(last_err)
    if not silent:
        st.error(f"❌ 캘린더 클라우드 저장 실패 (3회 재시도): {last_err}")
    return False
