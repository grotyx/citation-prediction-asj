# -*- coding: utf-8 -*-
"""Spine article citation-propensity predictor — Streamlit web app.

Run locally:  streamlit run app.py
"""
import base64
import html
import os
import tempfile

import streamlit as st

import scorer

BASE = os.path.dirname(os.path.abspath(__file__))
EMBLEM = os.path.join(BASE, "assets", "ksss_emblem.png")

st.set_page_config(page_title="Spine Citation Predictor", page_icon=EMBLEM, layout="centered")

# ---------------- styling: spine.or.kr visual identity ----------------
st.markdown(
    """<style>
@import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css");
:root{--navy:#293f94;--ink:#07192d;--soft:#6b7280;--line:#e5e7eb}
header[data-testid="stHeader"],footer,#MainMenu{display:none}
.block-container{max-width:960px;padding-top:0}
.ks-top{display:flex;align-items:center;gap:.75rem;padding:1.1rem 0;border-bottom:1px solid var(--line);margin-bottom:3rem}
.ks-top img{width:40px;height:40px}
.ks-top b{display:block;font-size:1.15rem;color:var(--ink);letter-spacing:-.01em}
.ks-top span{display:block;font-size:.68rem;letter-spacing:.06em;color:var(--soft)}
.ks-eyebrow{font-size:.8rem;font-weight:700;letter-spacing:.2em;color:var(--navy);margin:0 0 .6rem;text-transform:uppercase}
.ks-hero h1{font-size:2.9rem;line-height:1.18;font-weight:800;color:var(--ink);letter-spacing:-.03em;margin:0 0 1.8rem;padding:0}
.ks-mission{border-left:2px solid var(--navy);padding:.2rem 0 .2rem 1.1rem;margin-bottom:2.5rem}
.ks-mission p{margin:0 0 .35rem;color:#374151;line-height:1.7}
.ks-section{border-bottom:2px solid var(--navy);padding-bottom:.9rem;margin:2.2rem 0 1.4rem}
.ks-section h2{font-size:1.7rem;font-weight:700;color:var(--ink);margin:0;padding:0}
.ks-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media (max-width:640px){.ks-grid{grid-template-columns:1fr}.ks-hero h1{font-size:2.1rem}}
.ks-card{background:#fff;border:1px solid var(--line);border-radius:1rem;padding:1.4rem 1.5rem;box-shadow:0 1px 2px rgba(7,25,45,.04)}
.ks-card .num{font-size:3.2rem;font-weight:800;line-height:1;letter-spacing:-.03em;margin:.3rem 0 .5rem}
.ks-card .pill{display:inline-block;font-size:.8rem;font-weight:600;padding:.2rem .65rem;border-radius:9999px;color:#fff;vertical-align:middle;margin-left:.4rem;letter-spacing:0}
.ks-card .ask{font-size:1.05rem;color:var(--ink);margin:.2rem 0 0}
.ks-card .odds{font-size:1rem;font-weight:600;color:var(--ink);margin:0 0 .6rem}
.ks-card .odds span{font-weight:400;color:var(--soft)}
.ks-card .desc{font-size:.88rem;color:var(--soft);line-height:1.55;margin:0}
.ks-bar{position:relative;height:8px;border-radius:9999px;background:#eef0f6;margin:1rem 0 .35rem}
.ks-bar i{position:absolute;left:0;top:0;bottom:0;border-radius:9999px;background:var(--navy)}
.ks-bar u{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ink)}
.ks-scale{position:relative;display:flex;justify-content:space-between;font-size:.75rem;color:var(--soft)}
.ks-scale em{position:absolute;transform:translateX(-30%);font-style:normal;color:var(--ink);white-space:nowrap}
.ks-row{display:grid;grid-template-columns:15rem 1fr 4.5rem;align-items:center;gap:1rem;padding:.85rem 0;border-bottom:1px solid var(--line);font-size:.95rem;color:var(--ink)}
.ks-row .track{position:relative;height:8px;background:#eef0f6;border-radius:9999px}
.ks-row .track i{position:absolute;top:0;bottom:0;border-radius:9999px}
.ks-row .track u{position:absolute;left:50%;top:-4px;bottom:-4px;width:1px;background:#9ca3af}
.ks-row .v{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
@media (max-width:640px){.ks-row{grid-template-columns:1fr 4rem}.ks-row .track{display:none}}
.ks-note{border-left:2px solid var(--navy);background:#fff;padding:1rem 1.2rem;margin-top:1.6rem;color:#374151;line-height:1.65;font-size:.93rem}
.ks-foot{margin-top:4rem;padding-top:1.5rem;border-top:1px solid var(--line);font-size:.8rem;color:var(--soft);line-height:1.7}
.ks-foot a{color:var(--navy)}
.stTabs [data-baseweb="tab"]{font-weight:600;font-size:1rem}
.stButton button[kind="primary"]{height:3rem;font-weight:700;font-size:1rem}
</style>""",
    unsafe_allow_html=True,
)

# ---------------- header ----------------
with open(EMBLEM, "rb") as fh:
    emblem_b64 = base64.b64encode(fh.read()).decode()

st.markdown(
    f"<div class='ks-top'><img src='data:image/png;base64,{emblem_b64}' alt=''>"
    "<div><b>대한척추외과학회</b><span>KOREAN SOCIETY OF SPINE SURGERY</span></div></div>"
    "<div class='ks-hero'><p class='ks-eyebrow'>Spine Citation Predictor</p>"
    "<h1>How citable is<br>this spine article?</h1>"
    "<div class='ks-mission'><p class='ks-eyebrow'>About the score</p>"
    "<p>Estimates the probability that an article reaches the <b>top 25% of citations within its "
    "journal and year</b>, from its title, abstract, reference count and article type. The score is a "
    "surface-level correlate of <b>citability</b>. It does not measure scientific merit, methodological "
    "rigor, or clinical value, and it is <b>a research adjunct, not a basis for accept-or-reject "
    "decisions</b>.</p>"
    "<p style='font-size:.85rem;color:#6b7280'>Model: the reduced predictor set reported as primary in "
    "the study. Variables OpenAlex records only after publication (topic and subfield annotations, "
    "open-access status, current h-index) are excluded. Trained on 2018–2021, evaluated in 2022 and "
    "2023.</p></div></div>",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the model and text encoder… (first run takes ~30 s)")
def warm():
    scorer._models()
    scorer._encoder()
    return True


def _bar(p, base, top):
    """Horizontal bar for probability p on a 0..top scale, with a baseline marker."""
    w = min(p / top, 1.0) * 100
    b = base / top * 100
    return (
        f"<div class='ks-bar'><i style='width:{w:.1f}%'></i><u style='left:{b:.1f}%'></u></div>"
        f"<div class='ks-scale'><em style='left:{b:.1f}%'>Average {base*100:.0f}%</em>"
        f"<span style='margin-left:auto'>{top*100:.0f}%</span></div>"
    )


def _one_in(p):
    """Plain-language odds: 0.33 -> 'About 1 in 3', 0.7 -> 'About 7 in 10'."""
    if p >= 0.95:
        return "More than 9 in 10"
    if p >= 0.3:
        return f"About {round(p * 10)} in 10"
    return f"About 1 in {round(1 / p)}" if p > 0.005 else "Less than 1 in 200"


def render(res, low_conf=False, msid=None):
    if low_conf:
        st.warning(
            "The abstract or reference count was extracted with low confidence. "
            "Check the values above and score again."
        )
    p, pa = res["prob_injournal_top25"], res["prob_field_top10"]
    base_b, base_a = res["base_rate_top25"], res["base_rate_top10"]
    color = res["band_color"]
    sub = f" · {html.escape(msid)}" if msid else ""

    cards = (
        f"<div class='ks-section'><p class='ks-eyebrow'>Result{sub}</p>"
        "<h2>Predicted citation propensity</h2></div><div class='ks-grid'>"
        "<div class='ks-card'><p class='ks-eyebrow'>Primary · within the journal</p>"
        "<p class='ask'>Chance of reaching the <b>top 25%</b></p>"
        f"<div class='num' style='color:{color}'>{p*100:.0f}%"
        f"<span class='pill' style='background:{color}'>{res['band']}</span></div>"
        f"<p class='odds'>{_one_in(p)} <span>· an average article: 1 in 4</span></p>"
        "<p class='desc'>Top 25% by citations among articles published in the same journal in the "
        "same year. Such articles typically received about 5 citations (IQR 3–9) in the publication "
        "year and the next, versus about 1 (IQR 0–2) for other articles.</p>"
        f"{_bar(p, base_b, 1.0)}</div>"
        "<div class='ks-card'><p class='ks-eyebrow'>Secondary · whole spine field</p>"
        "<p class='ask'>Chance of reaching the <b>top 10%</b> (most-cited papers)</p>"
        f"<div class='num' style='color:#293f94'>{pa*100:.0f}%</div>"
        f"<p class='odds'>{_one_in(pa)} <span>· an average article: 1 in 10</span></p>"
        "<p class='desc'>Top 10% by citations among all articles in 13 spine journals in the same "
        "year, assuming publication in Asian Spine Journal. Such articles typically received about 9 "
        "citations (IQR 6–13) in the publication year and the next, versus about 2 (IQR 1–3) for "
        "other articles.</p>"
        f"{_bar(pa, base_a, 1.0)}</div></div>"
    )
    st.markdown(cards, unsafe_allow_html=True)

    drivers = [d for d in res["drivers"] if abs(d[1]) > 0.01]
    if drivers:
        rows = []
        for name, v in drivers:
            w = min(abs(v) / 0.15, 1.0) * 50  # half-track = 15 pp
            side = f"left:50%;width:{w:.1f}%" if v > 0 else f"right:50%;width:{w:.1f}%"
            fill = "#293f94" if v > 0 else "#cf222e"
            rows.append(
                f"<div class='ks-row'><span>{name}</span>"
                f"<div class='track'><u></u><i style='{side};background:{fill}'></i></div>"
                f"<span class='v' style='color:{fill}'>{v*100:+.0f} pp</span></div>"
            )
        st.markdown(
            "<div class='ks-section'><p class='ks-eyebrow'>Drivers</p><h2>What moved the score</h2></div>"
            "<p class='desc' style='color:#6b7280;font-size:.88rem;margin:-.6rem 0 .4rem'>Change in the "
            "predicted probability when one input is replaced by its cohort baseline. These are model "
            "sensitivities, not causal effects.</p>" + "".join(rows),
            unsafe_allow_html=True,
        )

    verdict = ("above the journal average" if p >= 0.30
               else "around the journal average" if p >= 0.18
               else "below the journal average")
    st.markdown(
        f"<div class='ks-note'>This article is predicted to be cited <b>{verdict}</b>. These are "
        "observational associations. The score is not a basis for accept-or-reject decisions, and "
        "padding reference lists, adjusting abstract length, or choosing an article type to raise it "
        "does not address what actually earns citations.</div>",
        unsafe_allow_html=True,
    )


tab1, tab2 = st.tabs(["Enter manually", "Upload a PDF"])

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
        st.caption("Detected from the text — untick if that is wrong.")

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
                "No Manuscript/Article Type field was found in the PDF. "
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

st.markdown(
    "<div class='ks-foot'>13 spine journals, 2018–2023 (n=13,299); trained on 2018–2021 and evaluated "
    "separately in 2022 and 2023. Primary reduced Model B ROC-AUC 0.721 (2022) and 0.706 (2023). The "
    "cohort contains only articles that were ultimately published, so performance in a submission pool "
    "containing rejected manuscripts is untested. Citation data from OpenAlex. This tool is a research "
    "adjunct and does not replace peer review. "
    "<a href='https://github.com/grotyx/citation-prediction-asj'>Code and data</a></div>",
    unsafe_allow_html=True,
)
