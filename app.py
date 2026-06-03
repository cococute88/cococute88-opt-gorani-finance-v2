import streamlit as st
import streamlit.components.v1 as components
from streamlit_oauth import OAuth2Component
import base64
import json
import pandas as pd
from streamlit.runtime.scriptrunner import StopException
from streamlit_cookies_manager import EncryptedCookieManager
from PIL import Image

try:
    icon_img = Image.open("bear.ico")
except FileNotFoundError:
    icon_img = "🧸"

st.set_page_config(
    page_title="Gorani finance",
    page_icon=icon_img,
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.caption("GORANI_FINANCE_APP_READY")

from core.sync import load_all_data  # noqa: E402
from core.firebase import load_data, save_data  # noqa: E402

MAX_FAVORITE_LINKS = 10

# =========================================================
# 💡 1. 즐겨찾기 링크 관리 기능 세팅 (팝업 + 표 편집)
# =========================================================
def normalize_url(url: str) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        return ""
    if not cleaned.startswith(("http://", "https://")):
        cleaned = f"https://{cleaned}"
    return cleaned

def load_favorite_links(user_id: str) -> list[dict]:
    # 💡 Firebase 경로 에러 방지: 이메일의 마침표(.)를 밑줄(_)로 치환합니다.
    safe_user_id = user_id.replace(".", "_")
    
    try:
        raw = load_data(safe_user_id, "favorite_links")
    except Exception:
        return []
        
    if not isinstance(raw, list):
        return []
        
    links = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        url = normalize_url(item.get("url") or "")
        if name and url:
            links.append({"name": name, "url": url})
        if len(links) >= MAX_FAVORITE_LINKS:
            break
    return links

def save_favorite_links(user_id: str, links: list[dict]) -> tuple[bool, str]:
    # 💡 Firebase 경로 에러 방지: 이메일의 마침표(.)를 밑줄(_)로 치환합니다.
    safe_user_id = user_id.replace(".", "_")
    
    cleaned_links = []
    for link in links[:MAX_FAVORITE_LINKS]:
        name = (link.get("name") or "").strip()
        url = normalize_url(link.get("url") or "")
        if name and url:
            cleaned_links.append({"name": name, "url": url})
            
    try:
        save_data(safe_user_id, "favorite_links", cleaned_links)
        return True, "✅ 저장 및 Firebase 동기화 완료!"
    except Exception as e:
        # 💡 진짜 에러 원인을 화면에 출력하도록 수정했습니다.
        return False, f"⚠️ 저장 실패 원인: {str(e)}"

def render_favorite_links_sidebar(user_id: str) -> None:
    state_key = "favorite_links"
    load_flag_key = "favorite_links_loaded"

    if not st.session_state.get(load_flag_key):
        st.session_state[state_key] = load_favorite_links(user_id)
        st.session_state[load_flag_key] = True

    links = st.session_state.get(state_key, [])

    with st.popover("⚙️ 즐겨찾는 링크 관리", use_container_width=True):
        st.caption(f"링크 이름과 주소를 입력하세요. (최대 {MAX_FAVORITE_LINKS}개)")
        
        df = pd.DataFrame(links, columns=["name", "url"])
        if df.empty:
            df = pd.DataFrame(columns=["name", "url"])
            
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "name": st.column_config.TextColumn("링크 이름", required=True, max_chars=20),
                "url": st.column_config.LinkColumn("URL 주소", required=True)
            },
            key="link_editor_component",
            use_container_width=True
        )
        
        if st.button("설정 저장하기", use_container_width=True, type="primary"):
            edited_df = edited_df.dropna(how="all")
            
            if len(edited_df) > MAX_FAVORITE_LINKS:
                st.error(f"최대 {MAX_FAVORITE_LINKS}개까지만 등록 가능합니다.")
            else:
                updated_links = []
                for _, row in edited_df.iterrows():
                    name = str(row.get("name", "")).strip()
                    url = normalize_url(str(row.get("url", "")))
                    if name and url:
                        updated_links.append({"name": name, "url": url})
                
                ok, msg = save_favorite_links(user_id, updated_links)
                if ok:
                    st.session_state[state_key] = updated_links
                    st.toast(msg, icon="✅")
                    st.rerun()
                else:
                    st.error(msg)

    if links:
        st.markdown("<div style='font-size: 13px; font-weight: bold; margin-top: 15px; margin-bottom: 5px;'>⭐ 즐겨찾는 링크</div>", unsafe_allow_html=True)
        for link in links:
            st.markdown(f"• [{link['name']}]({link['url']})")


# =========================================================
# 2. 구글 OAuth 및 기존 앱 기본 설정
# =========================================================
CLIENT_ID = st.secrets["google_oauth"]["client_id"]
CLIENT_SECRET = st.secrets["google_oauth"]["client_secret"]
REDIRECT_URI = st.secrets["google_oauth"]["redirect_uri"]
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_TOKEN_URL = "https://oauth2.googleapis.com/revoke"
oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, REVOKE_TOKEN_URL)

def _inject_global_layout_css() -> None:
    """Apply shared Streamlit layout spacing for all authenticated app pages."""
    st.markdown(
        """
        <style>
            /* 모든 페이지 메인 콘텐츠 상단 여백 축소 */
            main .block-container {
                padding-top: 0.75rem !important;
            }

            /* Streamlit 버전별 대응 */
            div[data-testid="stAppViewContainer"] main .block-container,
            div.block-container {
                padding-top: 0.75rem !important;
            }

            /* 페이지 제목 위쪽 여백 제거 */
            main h1,
            div.block-container h1 {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }

            /* 사이드바 상단 여백 축소 - 여러 Streamlit selector 대응 */
            section[data-testid="stSidebar"] div[data-testid="stSidebarContent"],
            section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
                padding-top: 0.5rem !important;
            }

            /* 사이드바 내부 첫 블록 여백 축소 */
            section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
                gap: 0.75rem !important;
            }

            /* 모바일에서는 상단 메뉴와 콘텐츠가 겹치지 않도록 약간의 여백 유지 */
            @media screen and (max-width: 768px) {
                main .block-container,
                div[data-testid="stAppViewContainer"] main .block-container,
                div.block-container {
                    padding-top: 0.5rem !important;
                }

                section[data-testid="stSidebar"] div[data-testid="stSidebarContent"],
                section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
                    padding-top: 0.25rem !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

components.html("""
    <script>
    var userAgent = navigator.userAgent.toLowerCase();
    if (userAgent.indexOf("kakaotalk") > -1) {
        var url = window.location.href;
        location.href = 'kakaotalk://web/openExternal?url=' + encodeURIComponent(url);
    }
    </script>
""", height=0)

cookies = EncryptedCookieManager(prefix="gorani", password=CLIENT_SECRET)

if not cookies.ready():
    st.info("🔄 자동 로그인 정보를 확인하고 있습니다. 잠시만 기다려주세요...")
    st.spinner("로딩 중...")
    st.stop()

is_authenticated = False
user_email = ""

saved_email = cookies.get("user_email")

if saved_email:
    user_email = saved_email
    is_authenticated = True

elif "token" in st.session_state:
    token = st.session_state["token"]
    id_token = token.get("id_token")
    payload = id_token.split(".")[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)
    user_info = json.loads(base64.b64decode(payload).decode("utf-8"))
    
    user_email = user_info.get("email")
    
    cookies["user_email"] = user_email
    cookies.save()
    is_authenticated = True

else:
    st.title("고라니 파이낸스🧸")
    st.write("안전한 데이터 관리를 위해 구글 로그인이 필요합니다.")
    
    result = oauth2.authorize_button(
        name="Google 계정으로 로그인",
        icon="https://www.google.com/favicon.ico",
        redirect_uri=REDIRECT_URI,
        scope="openid email profile",
        key="google_login"
    )
    
    if result and "token" in result:
        st.session_state["token"] = result.get("token")
        st.rerun()

# =========================================================
# 3. 인증 완료 시 메인 화면 및 즐겨찾기 렌더링
# =========================================================
if is_authenticated:
    user_uid = user_email
    
    if "user" not in st.session_state:
        st.session_state["user"] = {"uid": user_uid}
    
    load_all_data(user_uid)

    _inject_global_layout_css()

    pages = [
        st.Page("pages_app/1_asset_simulator.py", title="자산시뮬", icon="📊"),
        st.Page("pages_app/2_asset_tracker.py", title="자산트래커", icon="💰"),
        st.Page("pages_app/3_dividend_sim.py", title="양도세치기", icon="💵"),
        st.Page("pages_app/4_conversion_analysis.py", title="매도전환계산", icon="🔄"),
        st.Page("pages_app/5_dividend_calendar.py", title="배당캘린더", icon="📅", default=True),
        st.Page("pages_app/6_market_temperature.py", title="시장온도", icon="🌡️"),
        st.Page("pages_app/7_mdd_calculator.py", title="MDD계산", icon="📉"),
        st.Page("pages_app/8_attractiveness_score.py", title="SCHD매력도", icon="📈"),
        st.Page("pages_app/9_dividend_ledger.py", title="💵 배당금가계부"),
    ]

    nav = st.navigation(pages, position="top")

    try:
        nav.run()
    except StopException:
        pass

    with st.sidebar:
        st.divider()
        st.markdown(
            f"<div style='font-size: 13px; color: #6b7684; margin-bottom: 2px;'>"
            f"👤 <b>{user_email}</b>님"
            f"</div>", 
            unsafe_allow_html=True
        )
        text_col, btn_col = st.columns([8.5, 1.5])
        with text_col:
            st.markdown("<div style='font-size: 12px; color: #6b7684; padding-top: 8px;'>☁️ 자동저장 켜짐</div>", unsafe_allow_html=True)
        with btn_col:
            if st.button("❌", type="tertiary", help="로그아웃"):
                cookies["user_email"] = ""
                cookies.save()
                if "token" in st.session_state:
                    del st.session_state["token"]
                st.rerun()
        
        st.write("") 
        render_favorite_links_sidebar(user_uid)
