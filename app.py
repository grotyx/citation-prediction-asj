# -*- coding: utf-8 -*-
"""Spine manuscript citation-impact predictor — Streamlit web app.

Run locally:  streamlit run app.py
"""
import os
import tempfile
import streamlit as st

import scorer

st.set_page_config(page_title="Spine Citation Predictor", page_icon="📈", layout="centered")

# ---------------- header ----------------
st.title("📈 척추 논문 인용 영향력 예측")
st.caption("Citation propensity predictor for spine articles · 게재 기록 기반 대리변수 사용")

st.markdown(
    "제목·초록으로 **저널 내 인용 상위 25% 진입 확률**을 추정합니다. "
    "이 점수는 **인용 가능성(citability)** 의 표면적 상관지표이며, 과학적 가치·방법론적 엄밀성·임상적 유용성을 "
    "측정하지 않습니다. **채택/거절 결정 도구가 아닌 연구 단계 보조 지표**입니다."
)

st.info(
    "모델: 개정 논문의 primary 모델과 동일한 **reduced 예측변수 집합** (게재 후 부여되는 OpenAlex 주제·"
    "subfield 주석, open access 상태, 현재 h-index 제외). 학습 2018–2021 / 검증 2022·2023."
)


@st.cache_resource(show_spinner="모델·임베딩 로딩 중… (최초 1회, 수십 초 소요)")
def warm():
    scorer._models()
    scorer._encoder()
    return True


def render(res, low_conf=False, msid=None):
    if low_conf:
        st.warning("⚠️ PDF 추출 신뢰도가 낮습니다(초록/참고문헌). 아래 입력값을 확인·수정 후 다시 평가하세요.")
    p = res["prob_injournal_top25"]
    color = res["band_color"]

    st.markdown(f"### 결과{f' · {msid}' if msid else ''}")
    st.markdown(
        f"<div style='font-size:0.9rem;color:#555'>저널 내 인용 상위 25% 진입 확률</div>"
        f"<div style='font-size:2.6rem;font-weight:700;color:{color};line-height:1.1'>{p*100:.0f}%"
        f"<span style='font-size:1.1rem;margin-left:.5rem'>{res['band']}</span></div>"
        f"<div style='font-size:0.85rem;color:#777'>기준선 {res['base_rate_top25']*100:.0f}% (학습 코호트 실제 발생률) — 높을수록 평균 이상</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(1.0, p / 0.6))

    st.metric("분야 전체 상위 10% 확률 (secondary)", f"{res['prob_field_top10']*100:.0f}%",
              help=f"13개 척추 저널 전체 기준. 기준선 {res['base_rate_top10']*100:.0f}%. "
                   "이 모델은 저널 식별자를 사용하므로 판별력이 더 높습니다.")

    ups = [(d[0], d[1]) for d in res["drivers"] if d[1] > 0.01]
    dns = [(d[0], d[1]) for d in res["drivers"] if d[1] < -0.01]
    if ups or dns:
        st.markdown("#### 🔑 점수 영향 요인")
        st.caption("입력값을 코호트 기준값으로 바꿨을 때의 모델 민감도이며, 인과효과가 아닙니다.")
        if ups:
            st.markdown("⬆ **올림:** " + ", ".join(f"{n} (+{v*100:.0f}%p)" for n, v in ups))
        if dns:
            st.markdown("⬇ **내림:** " + ", ".join(f"{n} ({v*100:.0f}%p)" for n, v in dns))

    verdict = ("저널 평균보다 인용이 많을" if p >= 0.30
               else "저널 평균 수준으로 인용될" if p >= 0.18
               else "저널 평균보다 인용이 적을")
    st.info(f"📝 이 논문은 **{verdict} 가능성**으로 예측됩니다. "
            "(관찰적 상관에 근거한 보조 참고 지표 — 채택/거절 결정 도구 아니며, 점수를 올리기 위한 "
            "참고문헌 늘리기·초록 길이 조정·논문 유형 선택은 권장되지 않습니다.)")


tab1, tab2 = st.tabs(["✍️ 직접 입력", "📄 PDF 업로드"])

with tab1:
    title = st.text_input("제목 (Title)", placeholder="예: Contemporary outcomes after single-level lumbar fusion ...")
    abstract = st.text_area("초록 (Abstract)", height=200, placeholder="초록 전문을 붙여넣으세요 (영문).")
    c1, c2 = st.columns(2)
    n_refs = c1.number_input("참고문헌 수", min_value=0, max_value=300, value=30, step=1)

    if "review_touched" not in st.session_state:
        st.session_state.review_touched = False
    if not st.session_state.review_touched:
        st.session_state.is_review_cb = scorer.review_hint(title, abstract)

    def _mark_review_touched():
        st.session_state.review_touched = True

    is_review = c2.checkbox("리뷰/메타분석 논문", key="is_review_cb", on_change=_mark_review_touched)
    if not st.session_state.review_touched and st.session_state.is_review_cb:
        st.caption("💡 텍스트 기반 자동 감지 — 아니면 체크 해제하세요.")

    if st.button("평가하기", type="primary", use_container_width=True):
        if len((abstract or "").split()) < 20:
            st.error("초록을 20단어 이상 입력해 주세요 (예측 정확도를 위해).")
        else:
            warm()
            with st.spinner("평가 중…"):
                res = scorer.score(title, abstract, int(n_refs), is_review)
            render(res)

with tab2:
    up = st.file_uploader("원고 PDF 업로드", type=["pdf"])
    if up is not None:
        import pdf_extract
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(up.getbuffer())
            tmp = tf.name
        try:
            m = pdf_extract.extract_pdf(tmp)
        finally:
            os.unlink(tmp)
        st.success("추출 완료 — 값을 확인/수정한 뒤 평가하세요.")
        title2 = st.text_input("제목", value=m["title"], key="pt")
        abstract2 = st.text_area("초록", value=m["abstract"], height=180, key="pa")
        c1, c2 = st.columns(2)
        n2 = c1.number_input("참고문헌 수", min_value=0, max_value=300, value=int(m["n_references"]), step=1, key="pr")
        rev2 = c2.checkbox("리뷰/메타분석 논문", value=bool(m["is_review"]), key="prv")
        if not m["type_field_found"]:
            st.caption("⚠️ PDF에서 Manuscript/Article Type 필드를 찾지 못했습니다. 리뷰/메타분석 논문 여부를 직접 확인해 주세요.")
        if st.button("평가하기", type="primary", use_container_width=True, key="pbtn"):
            if len((abstract2 or "").split()) < 20:
                st.error("초록을 20단어 이상 입력해 주세요.")
            else:
                warm()
                with st.spinner("평가 중…"):
                    res = scorer.score(title2, abstract2, int(n2), rev2)
                render(res, low_conf=m["low_conf"], msid=m["msid"])

st.divider()
st.caption(
    "모델: 13개 척추 저널 2018–2023 (n=13,299), 학습 2018–2021 / 검증 2022·2023 각각. "
    "논문의 primary 모델(reduced) Model B ROC-AUC 0.721 (2022), 0.706 (2023). "
    "코호트는 최종 게재된 논문만 포함하며, 거절된 원고에서의 성능은 검증되지 않았습니다. "
    "인용 데이터 출처: OpenAlex. 본 도구는 연구 보조용이며 동료심사를 대체하지 않습니다."
)
