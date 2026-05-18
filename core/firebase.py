import streamlit as st
import firebase_admin
from firebase_admin import credentials, db

# 선생님의 Firebase 주소
BASE_URL = "https://myassettrackergorani-default-rtdb.firebaseio.com"

# 파이어베이스 앱이 아직 초기화되지 않았다면 '마스터키'로 초기화 진행
if not firebase_admin._apps:
    # Streamlit Secrets에 저장해둔 [firebase] 정보를 마스터키로 사용
    cred_dict = dict(st.secrets["firebase"])
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred, {
        'databaseURL': BASE_URL
    })

def save_data(uid, path, data):
    """마스터키 권한으로 파이어베이스에 데이터를 씁니다."""
    ref = db.reference(f'users/{uid}/{path}')
    ref.set(data)  # requests.put 대신 set() 사용

def load_data(uid, path):
    """마스터키 권한으로 파이어베이스에서 데이터를 읽어옵니다."""
    ref = db.reference(f'users/{uid}/{path}')
    return ref.get()  # requests.get 대신 get() 사용
