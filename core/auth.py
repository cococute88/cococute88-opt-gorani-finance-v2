import streamlit as st
import os
import json

# 자동 로그인을 기억할 비밀 수첩 파일 이름
TOKEN_FILE = ".login_token" 

def check_login():
    # 1. 이미 로그인해서 방에 들어와 있는 경우
    if "user" in st.session_state:
        return st.session_state["user"]
    
    # 2. 컴퓨터에 자동 로그인 수첩이 있는 경우 (재접속 시 자동 로그인!)
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            user = json.load(f)
            st.session_state["user"] = user
            return user

    # 3. 처음 왔거나 로그아웃 한 경우 (로그인 화면 띄우기)
    st.title("🔐 고라니 파이낸스 로그인")
    st.info("구글 이메일을 입력하면 선생님만의 전용 클라우드 공간이 만들어집니다.")
    
    email = st.text_input("구글 이메일 (예: gorani@gmail.com)")
    
    if st.button("구글 계정으로 시작 (자동 로그인 설정)", type="primary"):
        if email:
            # 이메일 특수문자를 언더바로 바꿔서 나만의 고유 ID(uid)로 만듦
            uid = email.replace("@", "_").replace(".", "_")
            user = {"uid": uid, "email": email}
            
            # 수첩에 적어서 다음부터는 묻지 않게 만듦
            with open(TOKEN_FILE, "w") as f:
                json.dump(user, f)
            
            st.session_state["user"] = user
            st.rerun() # 화면 새로고침해서 대문 통과!
    
    # 로그인 통과 전까지는 이 아래 코드를 절대 실행하지 않고 멈춤
    st.stop()

def logout():
    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE) # 수첩 찢어버리기 (자동로그인 해제)
    st.session_state.clear()
    st.rerun()