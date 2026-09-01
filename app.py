# -*- coding: utf-8 -*-
"""Spine article citation-propensity predictor — Streamlit web app.

Run locally:  streamlit run app.py
"""
import os
import tempfile

import streamlit as st

import scorer

st.set_page_config(page_title="Spine Citation Predictor", page_icon="📈", layout="centered")

# ---------------- header ----------------
st.title("📈 Spine Citation Predictor")
st.caption("Citation-propensity estimates for spine articles, from publication-record proxies")

st.markdown(
    "Estimates the probability that an article reaches the **top 25% of citations within its "
    "journal and year**, from its title, abstract, reference count and article type. The score is a "
    "surface-level correlate of **citability**. It does not measure scientific merit, "
    "methodological rigor, or clinical value, and it is **a research adjunct, not a basis for "
    "accept-or-reject decisions**."
)

st.info(
    "Model: the reduced predictor set reported as primary in the study — variables OpenAlex records "
    "only after publication (topic and subfield annotations, open-access status, current h-index) "
    "are excluded. Trained on 2018–2021, evaluated in 2022 and 2023."
)


@st.cache_resource(show_spinner="Loading the model and text encoder… (first run takes ~30 s)")
def warm():
    scorer._models()
    scorer._encoder()
    return True


def render(res, low_conf=False, msid=None):
    if low_conf:
        st.warning(
            "⚠️ The abstract or reference count was extracted with low confidence. "
            "Check the values above and score again."
        )
    p = res["prob_injournal_top25"]
    color = res["band_color"]

    st.markdown(f"### Result{f' · {msid}' if msid else ''}")
    st.markdown(
        f"<div style='font-size:0.9rem;color:#555'>Probability of within-journal top 25%</div>"
        f"<div style='font-size:2.6rem;font-weight:700;color:{color};line-height:1.1'>{p*100:.0f}%"
        f"<span style='font-size:1.1rem;margin-left:.5rem'>{res['band']}</span></div>"
        f"<div style='font-size:0.85rem;color:#777'>Baseline {res['base_rate_top25']*100:.0f}% "
        f"(observed rate in the training cohort)</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(1.0, p / 0.6))

    st.metric(
        "Probability of field-wide top 10% (secondary)",
        f"{res['prob_field_top10']*100:.0f}%",
        help=f"Across all 13 spine journals. Baseline {res['base_rate_top10']*100:.0f}%. "
             "This model also sees journal identity, which is why it discriminates better.",
    )

    ups = [(d[0], d[1]) for d in res["drivers"] if d[1] > 0.01]
    dns = [(d[0], d[1]) for d in res["drivers"] if d[1] < -0.01]
    if ups or dns:
        st.markdown("#### 🔑 What moved the score")
        st.caption(
            "Change in the predicted probability when one input is replaced by its cohort "
            "baseline. These are model sensitivities, not causal effects."
        )
        if ups:
            st.markdown("⬆ **Raised it:** " + ", ".join(f"{n} (+{v*100:.0f} pp)" for n, v in ups))
        if dns:
            st.markdown("⬇ **Lowered it:** " + ", ".join(f"{n} ({v*100:.0f} pp)" for n, v in dns))

    verdict = ("above the journal average" if p >= 0.30
               else "around the journal average" if p >= 0.18
               else "below the journal average")
    st.info(
        f"📝 This article is predicted to be cited **{verdict}**. These are observational "
        "associations. The score is not a basis for accept-or-reject decisions, and padding "
        "reference lists, adjusting abstract length, or choosing an article type to raise it does "
        "not address what actually earns citations."
    )


tab1, tab2 = st.tabs(["✍️ Enter manually", "📄 Upload a PDF"])

with tab1:
    title = st.text_input(
        "Title",
        placeholder="e.g. Contemporary outcomes after single-level lumbar fusion ...",
    )
    abstract = st.text_area(
        "Abstract", height=200, placeholder="Paste the full abstract (English)."
    )
    c1, c2 = st.columns(2)
    n_refs = c1.number_input("Number of references", min_value=0, max_value=300, value=30, step=1)

    if "review_touched" not in st.session_state:
        st.session_state.review_touched = False
    if not st.session_state.review_touched:
        st.session_state.is_review_cb = scorer.review_hint(title, abstract)

    def _mark_review_touched():
        st.session_state.review_touched = True

    is_review = c2.checkbox(
        "Review or meta-analysis", key="is_review_cb", on_change=_mark_review_touched
    )
    if not st.session_state.review_touched and st.session_state.is_review_cb:
        st.caption("💡 Detected from the text — untick if that is wrong.")

    if st.button("Score", type="primary", use_container_width=True):
        if len((abstract or "").split()) < 20:
            st.error("Please enter an abstract of at least 20 words.")
        else:
            warm()
            with st.spinner("Scoring…"):
                res = scorer.score(title, abstract, int(n_refs), is_review)
            render(res)

with tab2:
    up = st.file_uploader("Upload the manuscript PDF", type=["pdf"])
    if up is not None:
        import pdf_extract

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(up.getbuffer())
            tmp = tf.name
        try:
            m = pdf_extract.extract_pdf(tmp)
        finally:
            os.unlink(tmp)
        st.success("Extracted — check the values below, then score.")
        title2 = st.text_input("Title", value=m["title"], key="pt")
        abstract2 = st.text_area("Abstract", value=m["abstract"], height=180, key="pa")
        c1, c2 = st.columns(2)
        n2 = c1.number_input(
            "Number of references", min_value=0, max_value=300,
            value=int(m["n_references"]), step=1, key="pr",
        )
        rev2 = c2.checkbox("Review or meta-analysis", value=bool(m["is_review"]), key="prv")
        if not m["type_field_found"]:
            st.caption(
                "⚠️ No Manuscript/Article Type field was found in the PDF. "
                "Please set the review flag yourself."
            )
        if st.button("Score", type="primary", use_container_width=True, key="pbtn"):
            if len((abstract2 or "").split()) < 20:
                st.error("Please enter an abstract of at least 20 words.")
            else:
                warm()
                with st.spinner("Scoring…"):
                    res = scorer.score(title2, abstract2, int(n2), rev2)
                render(res, low_conf=m["low_conf"], msid=m["msid"])

st.divider()
st.caption(
    "13 spine journals, 2018–2023 (n=13,299); trained on 2018–2021 and evaluated separately in 2022 "
    "and 2023. Primary reduced Model B ROC-AUC 0.721 (2022) and 0.706 (2023). The cohort contains "
    "only articles that were ultimately published, so performance in a submission pool containing "
    "rejected manuscripts is untested. Citation data from OpenAlex. This tool is a research adjunct "
    "and does not replace peer review. "
    "[Code and data](https://github.com/grotyx/citation-prediction-asj)"
)
