import streamlit as st
import openai
import os

# 🔐 OpenAI API 키 설정
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="맞춤형 레시피 생성기", page_icon="🍳")
st.title("🍳 맞춤형 레시피 생성기")
st.write("사용할 재료를 입력하면, 그 재료로 만들 수 있는 요리와 단계별 레시피를 생성해드립니다.")

# 사용자 입력
ingredients = st.text_area("📝 사용하고 싶은 재료를 입력하세요 (쉼표로 구분)", placeholder="예: 계란, 우유, 밀가루, 설탕")

if st.button("레시피 생성하기") and ingredients.strip():
    with st.spinner("레시피를 생성 중입니다..."):

        # OpenAI에 보낼 프롬프트 생성
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
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )

            result = response["choices"][0]["message"]["content"]
            st.success("레시피 생성 완료!")
            st.markdown(result)

        except Exception as e:
            st.error(f"레시피 생성 중 오류가 발생했습니다: {str(e)}")
else:
    st.info("좌측에 재료를 입력한 후, '레시피 생성하기' 버튼을 눌러주세요.")
