import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# ─────────────────────────────────────────────
# Streamlit 앱 설정
# ─────────────────────────────────────────────
st.set_page_config(page_title="맞춤형 레시피 생성기", page_icon="🍳")
st.title("🍳 맞춤형 레시피 생성기")
st.write("입력한 재료로 만들 수 있는 요리와 단계별 레시피를 생성해드립니다.")

# ─────────────────────────────────────────────
# 로컬 모델 로딩 (최초 실행 시 시간이 좀 걸립니다)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_name = "tiiuae/falcon-7b-instruct"  # 또는 "gpt2", "mistralai/Mistral-7B-Instruct-v0.1" 등
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

text_generator = load_model()

# ─────────────────────────────────────────────
# 사용자 입력
# ─────────────────────────────────────────────
ingredients = st.text_area(
    "📝 사용하고 싶은 재료를 입력하세요 (쉼표로 구분)",
    placeholder="예: 계란, 우유, 밀가루, 설탕"
)

# ─────────────────────────────────────────────
# 레시피 생성 버튼
# ─────────────────────────────────────────────
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
            output = text_generator(prompt, max_new_tokens=300, temperature=0.7)[0]["generated_text"]
            # 프롬프트 이후 텍스트만 추출
            result = output[len(prompt):].strip()
            st.success("✅ 레시피 생성 완료!")
            st.markdown(result)

        except Exception as e:
            st.error(f"❌ 레시피 생성 중 오류 발생:\n\n{e}")
else:
    st.info("재료를 입력한 후, '레시피 생성하기' 버튼을 눌러주세요.")
