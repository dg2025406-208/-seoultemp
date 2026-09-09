import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

CUTOFF_YEAR = 2025
MIN_DAYS = 300


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    df = df[["날짜", "평균기온"]].copy()

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])

    df["연도"] = df["날짜"].dt.year

    return df


@st.cache_data
def make_yearly_data(df):
    yearly = (
        df[df["연도"] <= CUTOFF_YEAR]
        .groupby("연도")
        .agg(
            관측일수=("평균기온", "count"),
            연평균기온=("평균기온", "mean"),
        )
        .reset_index()
    )

    yearly = yearly[yearly["관측일수"] >= MIN_DAYS].copy()
    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


def make_regression(yearly):
    x = yearly["연도"].to_numpy(dtype=float)
    y = yearly["연평균기온"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    correlation = np.corrcoef(x, y)[0, 1]

    # 1년당 기온 변화량 → 100년당 기온 변화량
    slope_per_100_years = slope * 100

    return slope, intercept, slope_per_100_years, correlation


def predict_temperature(year, slope, intercept):
    return slope * year + intercept


st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온의 변화 추세를 선형 회귀로 분석합니다.")


# ---------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------

try:
    df = load_data()
    yearly = make_yearly_data(df)

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


if len(yearly) < 2:
    st.error("회귀 직선을 계산하기에 충분한 연도별 데이터가 없습니다.")
    st.stop()


# ---------------------------------------------------------
# 전체 기간 회귀
# ---------------------------------------------------------

(
    slope,
    intercept,
    slope_per_100,
    correlation,
) = make_regression(yearly)


start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
year_count = len(yearly)


# ---------------------------------------------------------
# 최근 20년 회귀
# ---------------------------------------------------------

recent_20 = yearly.tail(20).copy()

if len(recent_20) >= 2:
    (
        recent_slope,
        recent_intercept,
        recent_slope_per_100,
        recent_correlation,
    ) = make_regression(recent_20)

    recent_start_year = int(recent_20["연도"].min())
    recent_end_year = int(recent_20["연도"].max())
else:
    recent_slope = None
    recent_intercept = None
    recent_slope_per_100 = None
    recent_correlation = None
    recent_start_year = None
    recent_end_year = None


# ---------------------------------------------------------
# 핵심 지표: 100년에 몇 도 오르는가
# ---------------------------------------------------------

st.subheader("기온 상승 추세")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "전체 기간",
        f"{slope_per_100:+.2f} °C / 100년",
        help="회귀 직선의 기울기를 100년 기준으로 환산한 값입니다.",
    )
    st.caption(
        f"{start_year}~{end_year}년, {year_count}개 연도"
    )

with col2:
    if recent_slope_per_100 is not None:
        st.metric(
            "최근 20년",
            f"{recent_slope_per_100:+.2f} °C / 100년",
            help="회귀에 사용된 가장 최근 20개 연도를 기준으로 계산했습니다.",
        )
        st.caption(
            f"{recent_start_year}~{recent_end_year}년, "
            f"{len(recent_20)}개 연도"
        )


st.info(
    f"전체 기간의 연평균기온은 100년당 약 "
    f"**{slope_per_100:+.2f} °C** 변화하는 추세입니다."
)


# ---------------------------------------------------------
# 회귀 비교 상세
# ---------------------------------------------------------

st.subheader("전체 기간과 최근 20년 비교")

compare_col1, compare_col2 = st.columns(2)

with compare_col1:
    st.markdown("### 전체 기간")
    st.write(f"분석 기간: **{start_year}~{end_year}년**")
    st.write(f"사용 연도: **{year_count}개**")
    st.write(f"100년당 변화: **{slope_per_100:+.2f} °C**")
    st.write(f"상관계수: **{correlation:.4f}**")

with compare_col2:
    st.markdown("### 최근 20년")
    st.write(
        f"분석 기간: **{recent_start_year}~{recent_end_year}년**"
    )
    st.write(f"사용 연도: **{len(recent_20)}개**")
    st.write(f"100년당 변화: **{recent_slope_per_100:+.2f} °C**")
    st.write(f"상관계수: **{recent_correlation:.4f}**")


# ---------------------------------------------------------
# 연도 슬라이더
# ---------------------------------------------------------

st.subheader("예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)

predicted_temperature = predict_temperature(
    selected_year,
    slope,
    intercept,
)

st.markdown(
    f"""
    <div style="
        background-color: #f0f7ff;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin: 10px 0 30px 0;
    ">
        <div style="font-size: 20px; color: #555;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="
            font-size: 56px;
            font-weight: 700;
            color: #1976D2;
            margin-top: 8px;
        ">
            {predicted_temperature:.2f} °C
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 산점도 + 전체 기간 회귀 직선
# ---------------------------------------------------------

st.subheader("연도별 평균기온과 회귀 직선")

line_x = np.array([1900, 2100], dtype=float)
line_y = predict_temperature(line_x, slope, intercept)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7,
            color="#1976D2",
            opacity=0.75,
        ),
        customdata=yearly["관측일수"],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(
            color="#E53935",
            width=3,
        ),
        hovertemplate=(
            "연도: %{x}<br>"
            "회귀 예상값: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers",
        name=f"{selected_year}년 예측",
        marker=dict(
            size=14,
            color="#FF9800",
            line=dict(
                color="white",
                width=2,
            ),
        ),
        hovertemplate=(
            f"<b>{selected_year}년</b><br>"
            f"예상 연평균기온: {predicted_temperature:.2f} °C"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(range=[1900, 2100]),
    template="plotly_white",
    hovermode="closest",
    height=600,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# 회귀식
# ---------------------------------------------------------

st.subheader("회귀식")

st.write(
    f"전체 기간 회귀식: "
    f"**y = {slope:.6f} × 연도 + {intercept:.3f}**"
)

st.write(
    f"전체 기간 기울기: "
    f"**{slope_per_100:+.2f} °C / 100년**"
)

st.write(
    f"최근 20년 회귀식: "
    f"**y = {recent_slope:.6f} × 연도 + {recent_intercept:.3f}**"
)

st.write(
    f"최근 20년 기울기: "
    f"**{recent_slope_per_100:+.2f} °C / 100년**"
)


# ---------------------------------------------------------
# 사용된 데이터
# ---------------------------------------------------------

with st.expander("회귀에 사용된 연도별 데이터 보기"):
    display_df = yearly.copy()
    display_df["연평균기온"] = display_df["연평균기온"].map(
        lambda x: f"{x:.2f} °C"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


st.caption(
    "데이터 출처: 서울 기온 데이터(seoul.csv) | "
    "분석 기준 연도: 2025년까지 | "
    f"연간 관측일 {MIN_DAYS}일 미만인 연도 제외"
)
