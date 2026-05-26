"""Streamlit demo for polarization detection with Bayesian-optimized models."""

from __future__ import annotations

from html import escape
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.constants import DEMO_EXAMPLES, LABEL_DESCRIPTIONS, LABEL_NAMES, MODEL_FILES
from src.model_io import load_metadata, load_model
from src.predictor import predict_text

st.set_page_config(
    page_title="Polarization Detection Demo",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2.2rem;
    max-width: 1160px;
}
.main-header {
    padding: 1.35rem 1.5rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%);
    border: 1px solid #e5e7eb;
    margin-bottom: 1.15rem;
}
.main-header h1 {
    margin: 0 0 0.35rem 0;
    font-size: 2.05rem;
    line-height: 1.2;
    color: #111827;
}
.main-header p {
    margin: 0;
    color: #4b5563;
    font-size: 1.02rem;
    line-height: 1.6;
}
.tag-row {
    display: flex;
    gap: 0.55rem;
    flex-wrap: wrap;
    margin-top: 0.95rem;
}
.tag {
    padding: 0.28rem 0.72rem;
    border-radius: 999px;
    background: #ffffff;
    color: #1f2937;
    border: 1px solid #dbeafe;
    font-size: 0.88rem;
}
.section-title {
    font-weight: 700;
    font-size: 1.15rem;
    color: #111827;
    margin: 0.3rem 0 0.7rem 0;
}
.metric-card {
    padding: 0.85rem 0.95rem;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
}
.metric-label {
    color: #6b7280;
    font-size: 0.88rem;
    margin-bottom: 0.25rem;
}
.metric-value {
    color: #111827;
    font-size: 1.55rem;
    line-height: 1.2;
    font-weight: 750;
}
.result-card {
    padding: 1rem 1.1rem;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
}
.result-label {
    color: #6b7280;
    font-size: 0.88rem;
    margin-bottom: 0.25rem;
}
.result-value {
    color: #111827;
    font-size: 1.45rem;
    line-height: 1.3;
    font-weight: 750;
}
.prob-card {
    padding: 1rem 1.1rem;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
}
.prob-title {
    color: #374151;
    font-size: 0.95rem;
    margin-bottom: 0.2rem;
    font-weight: 600;
}
.prob-value {
    color: #111827;
    font-size: 1.85rem;
    line-height: 1.2;
    font-weight: 750;
}
.small-muted {
    color: #6b7280;
    font-size: 0.9rem;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stTextArea"] label {
    font-weight: 600;
}
div[data-testid="stExpander"] {
    margin-top: 0.65rem;
}
</style>
"""


@st.cache_resource(show_spinner="Đang tải mô hình...")
def get_cached_model(model_name: str) -> Any:
    return load_model(model_name)


@st.cache_data(show_spinner=False)
def get_cached_metadata() -> dict[str, Any]:
    return load_metadata()


def render_custom_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def format_metric(value: object) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return str(value)


def format_count(value: object) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, str) and value.isdigit():
        return f"{int(value):,}"
    return str(value)


def render_header() -> None:
    st.markdown(
        """
        <div class="main-header">
            <h1>Phát hiện phân cực quan điểm</h1>
            <p>
                Demo cho phép nhập một đoạn văn bản tiếng Anh và xem xác suất dự đoán
                của từng nhãn. Ba mô hình sử dụng đặc trưng TF-IDF và bộ siêu tham số
                đã được tối ưu bằng Bayesian Optimization.
            </p>
            <div class="tag-row">
                <span class="tag">English text</span>
                <span class="tag">TF-IDF</span>
                <span class="tag">Bayesian Optimization</span>
                <span class="tag">Binary classification</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(metadata: dict[str, Any]) -> str:
    st.sidebar.title("Cấu hình demo")
    model_name = st.sidebar.selectbox(
        "Chọn mô hình",
        options=list(MODEL_FILES.keys()),
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Ý nghĩa nhãn")
    for label, description in LABEL_DESCRIPTIONS.items():
        st.sidebar.markdown(f"**{label} - {LABEL_NAMES[label]}**")
        st.sidebar.caption(description)

    dataset = metadata.get("dataset", {})
    if dataset:
        counts = dataset.get("label_counts", {})
        st.sidebar.divider()
        st.sidebar.subheader("Dataset")
        st.sidebar.caption(dataset.get("name", "POLAR English dataset"))
        st.sidebar.write(f"Sau tiền xử lý: **{format_count(dataset.get('clean_rows', 'N/A'))}** mẫu")
        st.sidebar.write(f"Nhãn 0: **{format_count(counts.get('0', 'N/A'))}** mẫu")
        st.sidebar.write(f"Nhãn 1: **{format_count(counts.get('1', 'N/A'))}** mẫu")

    return model_name


def render_metric_card(label: str, value: object) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(format_metric(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_summary(model_name: str, metadata: dict[str, Any]) -> None:
    model_info = metadata.get("models", {}).get(model_name, {})
    if not model_info:
        return

    st.markdown('<div class="section-title">Hiệu suất thực nghiệm của mô hình</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Accuracy", model_info.get("accuracy", "N/A"))
    with c2:
        render_metric_card("Precision", model_info.get("precision_macro", "N/A"))
    with c3:
        render_metric_card("Recall", model_info.get("recall_macro", "N/A"))
    with c4:
        render_metric_card("F1-Macro", model_info.get("f1_macro", "N/A"))

    st.markdown('<div style="height: 0.45rem;"></div>', unsafe_allow_html=True)
    with st.expander("Thông tin mô hình"):
        st.write(f"**Phương pháp tối ưu:** {model_info.get('tuning_method', 'Bayesian Optimization')}")
        params = pd.DataFrame(
            [{"Tham số": key, "Giá trị": str(value)} for key, value in model_info.get("best_params", {}).items()]
        )
        if not params.empty:
            st.dataframe(params, use_container_width=True, hide_index=True)


def render_probability_cards(probabilities: dict[int, float]) -> None:
    cols = st.columns(2)
    for idx, label in enumerate([0, 1]):
        probability = probabilities.get(label, 0.0)
        with cols[idx]:
            st.markdown(
                f"""
                <div class="prob-card">
                    <div class="prob-title">Nhãn {label} - {LABEL_NAMES[label]}</div>
                    <div class="prob-value">{probability * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(float(probability))


def render_probability_chart(probabilities: dict[int, float]) -> None:
    chart_df = pd.DataFrame(
        {
            "Nhãn": [f"Nhãn {label}: {LABEL_NAMES[label]}" for label in [0, 1]],
            "Xác suất": [probabilities.get(label, 0.0) for label in [0, 1]],
        }
    )
    chart_df["Tỷ lệ"] = chart_df["Xác suất"].map(lambda value: f"{value * 100:.2f}%")

    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=10, size=38)
        .encode(
            y=alt.Y(
                "Nhãn:N",
                sort=None,
                title=None,
                axis=alt.Axis(labelAngle=0, labelFontSize=14, labelLimit=360),
            ),
            x=alt.X(
                "Xác suất:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title="Xác suất", labelFontSize=13, titleFontSize=14, grid=True),
            ),
            color=alt.Color(
                "Nhãn:N",
                legend=None,
                scale=alt.Scale(range=["#2563eb", "#f97316"]),
            ),
            tooltip=[
                alt.Tooltip("Nhãn:N", title="Nhãn"),
                alt.Tooltip("Tỷ lệ:N", title="Xác suất"),
            ],
        )
    )

    inside_labels = (
        alt.Chart(chart_df[chart_df["Xác suất"] >= 0.18])
        .mark_text(align="right", baseline="middle", dx=-10, fontSize=14, fontWeight="bold", color="#ffffff")
        .encode(
            y=alt.Y("Nhãn:N", sort=None, title=None),
            x=alt.X("Xác suất:Q", scale=alt.Scale(domain=[0, 1])),
            text="Tỷ lệ:N",
        )
    )

    outside_labels = (
        alt.Chart(chart_df[chart_df["Xác suất"] < 0.18])
        .mark_text(align="left", baseline="middle", dx=8, fontSize=14, fontWeight="bold", color="#111827")
        .encode(
            y=alt.Y("Nhãn:N", sort=None, title=None),
            x=alt.X("Xác suất:Q", scale=alt.Scale(domain=[0, 1])),
            text="Tỷ lệ:N",
        )
    )

    chart = (bars + inside_labels + outside_labels).properties(height=190).configure_view(strokeWidth=0)
    st.altair_chart(chart, use_container_width=True)


def render_prediction_result(model_name: str, user_text: str) -> None:
    model = get_cached_model(model_name)
    result = predict_text(model, user_text)

    st.divider()
    st.markdown('<div class="section-title">Kết quả dự đoán</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Mô hình đang sử dụng</div>
            <div class="result-value">{escape(model_name)}</div>
            <div style="height: 0.75rem;"></div>
            <div class="result-label">Nhãn dự đoán</div>
            <div class="result-value">Nhãn {result.predicted_label} - {escape(result.predicted_name)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    render_probability_cards(result.probabilities)
    st.write("")
    render_probability_chart(result.probabilities)

    with st.expander("Chi tiết tiền xử lý"):
        compare_df = pd.DataFrame(
            [
                {"Loại": "Raw text", "Nội dung": result.raw_text},
                {"Loại": "Cleaned text", "Nội dung": result.cleaned_text or "<empty>"},
            ]
        )
        st.dataframe(compare_df, use_container_width=True, hide_index=True)


def render_input_area() -> str:
    st.divider()
    st.markdown('<div class="section-title">Nhập văn bản cần phân tích</div>', unsafe_allow_html=True)

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    if "last_example" not in st.session_state:
        st.session_state.last_example = "Tự nhập"

    selected_example = st.selectbox("Chọn câu mẫu demo", options=list(DEMO_EXAMPLES.keys()))
    if selected_example != st.session_state.last_example:
        st.session_state.input_text = DEMO_EXAMPLES[selected_example]["text"]
        st.session_state.last_example = selected_example

    return st.text_area(
        "Text input",
        height=150,
        placeholder="Nhập hoặc dán một đoạn văn bản tiếng Anh tại đây...",
        key="input_text",
        label_visibility="collapsed",
    )


def main() -> None:
    render_custom_css()
    metadata = get_cached_metadata()
    model_name = render_sidebar(metadata)

    render_header()
    render_model_summary(model_name, metadata)

    user_text = render_input_area()
    analyze = st.button("Phân tích", type="primary", use_container_width=True)

    if analyze:
        if not user_text.strip():
            st.error("Vui lòng nhập văn bản trước khi phân tích.")
            return
        render_prediction_result(model_name, user_text)


if __name__ == "__main__":
    main()
