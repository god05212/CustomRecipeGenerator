import os
import streamlit as st
import openai

# ─────────────────────────────────────────────────────────
# API 키 입력 및 설정
# ─────────────────────────────────────────────────────────
# 사용자로부터 API 키 입력 받기
api_key = st.text_input("🔑 OpenAI API Key를 입력하세요:", type="password")

# API 키가 입력되었는지 확인
if not api_key:
    st.warning("OpenAI API 키를 입력해주세요.")
    st.stop()  # API 키 없으면 앱 실행 중지

# OpenAI API 키 설정
openai.api_key = api_key

# Streamlit 앱 설정
st.set_page_config(page_title="맞춤형 레시피 생성기", page_icon="🍳")
st.title("🍳 맞춤형 레시피 생성기")
st.write("입력한 재료로 만들 수 있는 요리와 단계별 레시피를 생성해드립니다.")

# 사용자 입력
ingredients = st.text_area(
    "📝 사용하고 싶은 재료를 입력하세요 (쉼표로 구분)",
    placeholder="예: 계란, 우유, 밀가루, 설탕"
)

if st.button("레시피 생성하기") and ingredients.strip():
    with st.spinner("레시피를 생성 중입니다..."):

        prompt = f"""
        다음 재료들을 사용해서 만들 수 있는 간단한 요리 하나를 추천해줘.
        그리고 그 요리의 이름과, 각 단계가 분리된 조리법을 상세히 알려줘.
        형식은 아래와 같게 해줘:

        요리 이름: <요리명>

        재료: <사용 재료 목록>

        조리법:
        1. ...
        2. ...
        3. ...

        재료 목록: {ingredients}
        """

        try:
            # 최신 방식으로 API 호출 (ChatCompletion 사용)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # gpt-3.5-turbo 또는 다른 모델 선택 가능
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )

            result = response['choices'][0]['message']['content'].strip()  # 응답 처리
            st.success("✅ 레시피 생성 완료!")
            st.markdown(result)

        except Exception as e:
            st.error(f"❌ 레시피 생성 중 오류 발생:\n\n{e}")
else:
    st.info("좌측에 재료를 입력한 후, '레시피 생성하기' 버튼을 눌러주세요.")
