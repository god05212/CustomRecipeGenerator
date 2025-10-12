import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="맞춤형 레시피 생성기", page_icon="🍳")
st.title("🍳 맞춤형 레시피 생성기")
st.write("입력한 재료로 만들 수 있는 요리와 단계별 레시피를 생성해드립니다.")

@st.cache_resource
def load_model():
    model_name = "skt/kogpt2-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

text_generator = load_model()

ingredients = st.text_area(
    "📝 사용하고 싶은 재료를 입력하세요 (쉼표로 구분)",
    placeholder="예: 계란, 우유, 밀가루"
)

if st.button("레시피 생성하기") and ingredients.strip():
    with st.spinner("레시피를 생성 중입니다..."):
        prompt = f"""
다음 재료들로 만들 수 있는 간단한 요리를 추천해줘.
요리 이름, 재료 목록, 그리고 단계별 조리법을 자세히 알려줘.

재료: {ingredients}
"""

        try:
            output = text_generator(prompt, max_new_tokens=300, temperature=0.7)[0]["generated_text"]
            # prompt 부분 제외
            result = output[len(prompt):].strip()
            st.success("✅ 레시피 생성 완료!")
            st.markdown(result)
        except Exception as e:
            st.error(f"❌ 오류 발생:\n\n{e}")
else:
    st.info("재료를 입력하고 '레시피 생성하기' 버튼을 눌러주세요.")
