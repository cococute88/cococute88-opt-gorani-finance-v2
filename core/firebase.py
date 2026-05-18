import streamlit as st
import firebase_admin
from firebase_admin import credentials, db


def _get_database_url():
    try:
        return st.secrets["firebase_app"]["database_url"]
    except KeyError as exc:
        raise RuntimeError(
            "Missing Streamlit secret: [firebase_app].database_url. "
            "Please set your v2 Firebase Realtime Database URL in secrets.toml."
        ) from exc


@st.cache_resource(show_spinner=False)
def _get_db_client():
    base_url = _get_database_url()
    if not firebase_admin._apps:
        cred_dict = dict(st.secrets["firebase"])
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred, {
            "databaseURL": base_url
        })
    return db


def save_data(uid, path, data):
    """마스터키 권한으로 파이어베이스에 데이터를 씁니다."""
    ref = _get_db_client().reference(f"users/{uid}/{path}")
    ref.set(data)


def load_data(uid, path):
    """마스터키 권한으로 파이어베이스에서 데이터를 읽어옵니다."""
    ref = _get_db_client().reference(f"users/{uid}/{path}")
    return ref.get()
