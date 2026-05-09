"""
🌿 원예장비 제조업체 총괄생산계획(APP) 최적화 대시보드
   강의록 setup_model 구조 기반 · Pyomo LP / Rounding / IP
   Author: [본인 이름] | Hongik University
"""

import math, itertools
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, SolverFactory,
    NonNegativeReals, NonNegativeIntegers, minimize, value,
)

# ══════════════════════════════════════════════════════════════
# 0. 페이지 설정
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="총괄생산계획 최적화",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 커스텀 CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f8f9fa; border-radius: 10px;
    padding: 14px 18px; border-left: 4px solid #4C78A8;
    margin-bottom: 8px;
  }
  .metric-card h4 { margin:0; font-size:12px; color:#6c757d; }
  .metric-card p  { margin:0; font-size:22px; font-weight:700; color:#212529; }
  .step-header {
    background: linear-gradient(90deg,#4C78A8 0%,#6ea8d8 100%);
    color: white; padding: 10px 18px; border-radius: 8px; margin-bottom:12px;
  }
  .step-header h3 { margin:0; font-size:16px; }
  .gap-box {
    border: 2px solid #e45756; border-radius:8px;
    padding:12px; background:#fff5f5; margin:8px 0;
  }
  .ok-box {
    border: 2px solid #54a24b; border-radius:8px;
    padding:12px; background:#f5fff5; margin:8px 0;
  }
  div[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 1. Pyomo 모델 (강의록 setup_model 구조 그대로)
# ══════════════════════════════════════════════════════════════

def setup_model(D: list, p: dict, type_mp: str = "LP") -> ConcreteModel:
    """
    강의록 5페이지 setup_model 함수.
    결정변수: W, H, L, P, I, S, C, O  (t = 0..TH)
    목적함수: Z = Σ(640W + 6O + 300H + 500L + 2I + 5S + 10P + 30C)
    """
    TH   = len(D)
    TIME = range(0, TH + 1)
    T    = range(1, TH + 1)
    tv   = NonNegativeIntegers if type_mp == "IP" else NonNegativeReals

    m = ConcreteModel()
    for name in ("W","H","L","P","I","S","C","O"):
        setattr(m, name, Var(TIME, domain=tv, bounds=(0, None)))

    rpc     = p["wage_reg"] * p["work_hrs"] * p["work_days"]   # 640
    cap_reg = p["work_hrs"] * p["work_days"] / p["std_time"]   # 40
    cap_ot  = 1.0 / p["std_time"]                              # 0.25

    # 목적함수
    m.Cost = Objective(
        expr=sum(
            rpc*m.W[t] + p["wage_ot"]*m.O[t]
            + p["cost_hire"]*m.H[t] + p["cost_fire"]*m.L[t]
            + p["cost_hold"]*m.I[t] + p["cost_short"]*m.S[t]
            + p["cost_mat"]*m.P[t]  + p["cost_sub"]*m.C[t]
            for t in T),
        sense=minimize)

    # 제약조건
    m.labor     = Constraint(T, rule=lambda m,t: m.W[t]==m.W[t-1]+m.H[t]-m.L[t])
    m.capacity  = Constraint(T, rule=lambda m,t: m.P[t]<=cap_reg*m.W[t]+cap_ot*m.O[t])
    m.inventory = Constraint(T, rule=lambda m,t:
        m.I[t]==m.I[t-1]+m.P[t]+m.C[t]-D[t-1]-m.S[t-1]+m.S[t])
    m.overtime  = Constraint(T, rule=lambda m,t: m.O[t]<=p["max_ot"]*m.W[t])
    m.W_0       = Constraint(rule=lambda m: m.W[0]==p["W0"])
    m.I_0       = Constraint(rule=lambda m: m.I[0]==p["I0"])
    m.S_0       = Constraint(rule=lambda m: m.S[0]==0)
    m.last_inv  = Constraint(rule=lambda m: m.I[TH]>=p["I_final"])
    m.last_sht  = Constraint(rule=lambda m: m.S[TH]==0)
    return m


def _extract(model, D, p, mtype):
    TH  = len(D); T = range(1, TH+1)
    rpc = p["wage_reg"]*p["work_hrs"]*p["work_days"]
    rows = []
    for t in T:
        W=value(model.W[t]); H=value(model.H[t]); L=value(model.L[t])
        Pv=value(model.P[t]); Iv=value(model.I[t]); Sv=value(model.S[t])
        Cv=value(model.C[t]); Ov=value(model.O[t])
        cr=rpc*W; co=p["wage_ot"]*Ov; ch=p["cost_hire"]*H; cf=p["cost_fire"]*L
        ci=p["cost_hold"]*Iv; cs=p["cost_short"]*Sv
        cm=p["cost_mat"]*Pv; cb=p["cost_sub"]*Cv
        rows.append({"월":f"{t}월","수요":D[t-1],
            "W":W,"H":H,"L":L,"P":Pv,"I":Iv,"S":Sv,"C":Cv,"O":Ov,
            "c_reg":cr,"c_ot":co,"c_hire":ch,"c_fire":cf,
            "c_hold":ci,"c_short":cs,"c_mat":cm,"c_sub":cb,
            "월비용":cr+co+ch+cf+ci+cs+cm+cb})
    df   = pd.DataFrame(rows)
    cost = round(value(model.Cost), 2)
    return {"df":df,"cost":cost,"type":mtype}


def run_model(D, p, type_mp="LP"):
    mdl    = setup_model(D, p, type_mp)
    solver = SolverFactory("glpk")
    res    = solver.solve(mdl, tee=False)
    status = str(res.solver.termination_condition)
    if status != "optimal":
        return None, status
    return _extract(mdl, D, p, type_mp), "optimal"


def simulate_rounding(D, p, W_int: list):
    """W를 정수로 고정 후 나머지 변수 재LP 최적화"""
    TH = len(D); T = range(1,TH+1)
    W0 = p["W0"]
    W_all = [W0]+W_int
    H = [max(0, W_all[t]-W_all[t-1]) for t in range(1,TH+1)]
    L = [max(0, W_all[t-1]-W_all[t]) for t in range(1,TH+1)]

    cap_reg = p["work_hrs"]*p["work_days"]/p["std_time"]
    cap_ot  = 1.0/p["std_time"]
    rpc     = p["wage_reg"]*p["work_hrs"]*p["work_days"]
    I0      = p["I0"]; If = p["I_final"]

    m = ConcreteModel()
    TIME = range(0, TH+1)
    for nm in ("P","I","S","C","O"):
        setattr(m, nm, Var(TIME, domain=NonNegativeReals, bounds=(0,None)))

    fixed = sum(rpc*W_all[t]+p["cost_hire"]*H[t-1]+p["cost_fire"]*L[t-1]
                for t in range(1,TH+1))
    m.Cost = Objective(
        expr=fixed+sum(p["wage_ot"]*m.O[t]+p["cost_hold"]*m.I[t]+
                       p["cost_short"]*m.S[t]+p["cost_mat"]*m.P[t]+p["cost_sub"]*m.C[t]
                       for t in T),
        sense=minimize)
    m.cap  = Constraint(T, rule=lambda m,t: m.P[t]<=cap_reg*W_all[t]+cap_ot*m.O[t])
    m.inv  = Constraint(T, rule=lambda m,t:
        m.I[t]==m.I[t-1]+m.P[t]+m.C[t]-D[t-1]-m.S[t-1]+m.S[t])
    m.ovt  = Constraint(T, rule=lambda m,t: m.O[t]<=p["max_ot"]*W_all[t])
    m.I_0  = Constraint(rule=lambda m: m.I[0]==I0)
    m.S_0  = Constraint(rule=lambda m: m.S[0]==0)
    m.Ilast= Constraint(rule=lambda m: m.I[TH]>=If)
    m.Slast= Constraint(rule=lambda m: m.S[TH]==0)

    solver = SolverFactory("glpk")
    res    = solver.solve(m, tee=False)
    status = str(res.solver.termination_condition)
    if status != "optimal":
        return None, status

    rows = []
    for t in T:
        W_=W_all[t]; H_=H[t-1]; L_=L[t-1]
        Pv=value(m.P[t]); Iv=value(m.I[t]); Sv=value(m.S[t])
        Cv=value(m.C[t]); Ov=value(m.O[t])
        cr=rpc*W_; ch=p["cost_hire"]*H_; cf=p["cost_fire"]*L_
        co=p["wage_ot"]*Ov; ci=p["cost_hold"]*Iv
        cs=p["cost_short"]*Sv; cm=p["cost_mat"]*Pv; cb=p["cost_sub"]*Cv
        rows.append({"월":f"{t}월","수요":D[t-1],
            "W":W_,"H":H_,"L":L_,"P":Pv,"I":Iv,"S":Sv,"C":Cv,"O":Ov,
            "c_reg":cr,"c_ot":co,"c_hire":ch,"c_fire":cf,
            "c_hold":ci,"c_short":cs,"c_mat":cm,"c_sub":cb,
            "월비용":cr+co+ch+cf+ci+cs+cm+cb})
    df   = pd.DataFrame(rows)
    cost = round(value(m.Cost), 2)
    return {"df":df,"cost":cost,"type":"Rounding"}, "optimal"


# ══════════════════════════════════════════════════════════════
# 2. 시각화 헬퍼
# ══════════════════════════════════════════════════════════════
COST_META = [
    ("c_reg",  "정규노동비", "#4C78A8"),
    ("c_ot",   "잔업비",     "#F58518"),
    ("c_hire", "고용비",     "#54A24B"),
    ("c_fire", "해고비",     "#E45756"),
    ("c_hold", "재고유지비", "#72B7B2"),
    ("c_short","부족재고비", "#B279A2"),
    ("c_mat",  "재료비",     "#FF9DA6"),
    ("c_sub",  "하청비",     "#9D755D"),
]
MODEL_CLR = {"LP":"#4C78A8","Rounding":"#F58518","IP":"#54A24B"}

LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=60,b=40,l=50,r=20),
    legend=dict(orientation="h", y=1.18, font_size=11),
    font=dict(size=12),
)

def _fig(**kw):
    fig = go.Figure()
    fig.update_layout(**LAYOUT, **kw)
    return fig


# ── 생산·수요 차트 ─────────────────────────────
def fig_production(results, D):
    months = results[0]["df"]["월"].tolist()
    fig = _fig(title="생산량 / 외주 vs 수요", barmode="group",
               xaxis_title="월", yaxis_title="수량 (ea)")
    for r in results:
        c = MODEL_CLR.get(r["type"],"#888")
        fig.add_bar(name=f"{r['type']} 생산", x=months, y=r["df"]["P"],
                    marker_color=c, opacity=0.8, legendgroup=r["type"])
        if r["df"]["C"].sum() > 0.1:
            fig.add_bar(name=f"{r['type']} 외주", x=months, y=r["df"]["C"],
                        marker_color=c, opacity=0.4, pattern_shape="/",
                        legendgroup=r["type"])
    fig.add_scatter(name="수요", x=months, y=D, mode="lines+markers",
                    line=dict(color="#E45756",width=2.5), marker=dict(size=7))
    return fig


# ── 재고 차트 ──────────────────────────────────
def fig_inventory(results, I_final):
    months = results[0]["df"]["월"].tolist()
    fig = _fig(title="재고 / 부족재고 추이", xaxis_title="월", yaxis_title="수량 (ea)")
    for r in results:
        c = MODEL_CLR.get(r["type"],"#888")
        fig.add_scatter(name=f"{r['type']} 재고", x=months, y=r["df"]["I"],
                        mode="lines+markers", line=dict(color=c,width=2),
                        fill="tozeroy", fillcolor=f"rgba({_hex2rgb(c)},0.1)")
        if r["df"]["S"].max() > 0.01:
            fig.add_scatter(name=f"{r['type']} 부족재고", x=months, y=r["df"]["S"],
                            mode="lines+markers", line=dict(color=c,dash="dash",width=1.5),
                            marker=dict(size=5,symbol="x"))
    fig.add_hline(y=I_final, line_dash="dot", line_color="gray",
                  annotation_text=f"최종재고 하한 {I_final}개",
                  annotation_position="bottom right")
    return fig


def _hex2rgb(h):
    h=h.lstrip("#"); return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


# ── 인력 차트 ──────────────────────────────────
def fig_workforce(results):
    months = results[0]["df"]["월"].tolist()
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    for r in results:
        c = MODEL_CLR.get(r["type"],"#888")
        fig.add_scatter(name=f"{r['type']} 작업자(W)", x=months, y=r["df"]["W"],
                        mode="lines+markers", line=dict(color=c,width=2.5),
                        marker=dict(size=7), secondary_y=False)
    # 고용·해고는 첫 번째 결과만 표시(대표)
    ref = results[0]; c0 = MODEL_CLR.get(ref["type"],"#888")
    fig.add_bar(name="고용(H)", x=months, y=ref["df"]["H"],
                marker_color="#54A24B", opacity=0.6, secondary_y=True)
    fig.add_bar(name="해고(L)", x=months, y=ref["df"]["L"],
                marker_color="#E45756", opacity=0.6, secondary_y=True)
    fig.update_layout(**LAYOUT, barmode="group",
                      title="인력 계획 (작업자 수 / 고용 / 해고)")
    fig.update_yaxes(title_text="작업자 수 (명)", secondary_y=False)
    fig.update_yaxes(title_text="고용·해고 (명)", secondary_y=True)
    return fig


# ── 잔업 차트 ──────────────────────────────────
def fig_overtime(results):
    months = results[0]["df"]["월"].tolist()
    fig = _fig(title="월별 초과근무 시간 (hr/월)", xaxis_title="월",
               yaxis_title="초과시간 (hr)")
    for r in results:
        c = MODEL_CLR.get(r["type"],"#888")
        fig.add_bar(name=f"{r['type']} 잔업", x=months, y=r["df"]["O"],
                    marker_color=c, opacity=0.8)
    fig.update_layout(barmode="group")
    return fig


# ── 비용 스택 차트 ─────────────────────────────
def fig_cost_stack(res):
    df = res["df"]; months = df["월"].tolist()
    fig = _fig(title=f"{res['type']} — 월별 비용 구성 (천원)",
               barmode="stack", xaxis_title="월", yaxis_title="비용 (천원)")
    for key, lbl, clr in COST_META:
        fig.add_bar(name=lbl, x=months, y=df[key], marker_color=clr, opacity=0.88)
    return fig


# ── 비용 파이 차트 ─────────────────────────────
def fig_cost_pie(res):
    df = res["df"]
    lbls  = [m[1] for m in COST_META]
    vals  = [df[m[0]].sum() for m in COST_META]
    clrs  = [m[2] for m in COST_META]
    fig = go.Figure(go.Pie(labels=lbls, values=vals, marker_colors=clrs,
                           hole=0.42, textinfo="label+percent",
                           sort=True, direction="clockwise"))
    fig.update_layout(**LAYOUT, title=f"{res['type']} — 비용 비중")
    return fig


# ── Duality Gap 바 차트 ────────────────────────
def fig_gap_bar(lp_c, rnd_c, ip_c):
    items = [("LP\n(이론 하한)", lp_c, MODEL_CLR["LP"]),
             ("반올림\n(실무 근사)", rnd_c, MODEL_CLR["Rounding"]),
             ("IP\n(정수 최적해)", ip_c, MODEL_CLR["IP"])]
    fig = _fig(title="Duality Gap — LP · 반올림 · IP 비용 비교",
               xaxis_title="모델", yaxis_title="총비용 (천원)")
    fig.add_bar(x=[i[0] for i in items], y=[i[1] for i in items],
                marker_color=[i[2] for i in items],
                text=[f"₩{i[1]:,.0f}" for i in items],
                textposition="outside", width=0.5)
    fig.add_hline(y=lp_c, line_dash="dash", line_color=MODEL_CLR["LP"],
                  annotation_text="LP 하한", annotation_position="right")
    return fig


# ── Waterfall 차트 ─────────────────────────────
def fig_waterfall(lp_c, rnd_c, ip_c):
    g1 = rnd_c - lp_c
    g2 = ip_c  - rnd_c
    fig = go.Figure(go.Waterfall(
        measure=["absolute","relative","relative"],
        x=["LP (하한)", f"반올림 Gap\n+{g1:,.0f}천", f"IP Gap\n+{g2:,.0f}천"],
        y=[lp_c, g1, g2],
        connector={"line":{"color":"rgba(0,0,0,0.2)"}},
        increasing={"marker":{"color":"#E45756"}},
        totals={"marker":{"color":"#4C78A8"}},
        text=[f"₩{lp_c:,.0f}",f"+₩{g1:,.0f}",f"+₩{g2:,.0f}"],
        textposition="outside",
    ))
    fig.update_layout(**LAYOUT, title="비용 증가 분해 (폭포 차트)",
                      yaxis_title="비용 (천원)", height=300)
    return fig


# ── 민감도 분석 차트 ───────────────────────────
def fig_sensitivity(base_cost, param_key, param_label, D, p_base):
    ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    base_v = p_base[param_key]
    costs  = []
    for r in ratios:
        p_tmp = {**p_base, param_key: base_v * r}
        res, st_ = run_model(D, p_tmp, "LP")
        costs.append(res["cost"] if res else None)
    x_labels = [f"×{r}" for r in ratios]
    fig = _fig(title=f"민감도 분석 — {param_label}",
               xaxis_title="파라미터 배율", yaxis_title="총비용 (천원)")
    fig.add_scatter(x=x_labels, y=costs, mode="lines+markers+text",
                    line=dict(color="#4C78A8",width=2.5), marker=dict(size=8),
                    text=[f"{c:,.0f}" if c else "—" for c in costs],
                    textposition="top center")
    fig.add_vline(x="×1.0", line_dash="dash", line_color="gray",
                  annotation_text="기준값")
    return fig


# ══════════════════════════════════════════════════════════════
# 3. 사이드바
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Hongik_University_logo.svg/200px-Hongik_University_logo.svg.png",
             width=110)
    st.markdown("### ⚙️ 파라미터 설정")

    # 수요
    with st.expander("📦 예상수요 (개/월)", expanded=True):
        d_str = st.text_input("쉼표로 구분", "1600, 3000, 3200, 3800, 2200, 2200",
                              help="예: 1600, 3000, 3200, 3800, 2200, 2200")
        try:
            D = [int(x.strip()) for x in d_str.split(",")]
            assert all(v>=0 for v in D) and len(D)>=2
            st.success(f"{len(D)}개월 계획 | 총 {sum(D):,}개")
        except:
            st.error("올바른 정수를 쉼표로 입력하세요.")
            D = [1600,3000,3200,3800,2200,2200]

    with st.expander("👷 초기 조건"):
        W0      = st.number_input("초기 종업원 수 (명)", 0, 500, 80, 1)
        I0      = st.number_input("초기 재고 (개)",      0, 10000, 1000, 100)
        I_final = st.number_input("최종 재고 하한 (개)", 0, 10000, 500,  100)

    with st.expander("💰 노동 비용 (천원)"):
        wage_reg   = st.number_input("정규임금 /시간",    0.0, 100.0, 4.0,  0.5)
        wage_ot    = st.number_input("초과임금 /시간",    0.0, 100.0, 6.0,  0.5)
        cost_hire  = st.number_input("고용비 /인",        0,   5000,  300,  50)
        cost_fire  = st.number_input("해고비 /인",        0,   5000,  500,  50)

    with st.expander("📦 재고·외주 비용 (천원)"):
        cost_hold  = st.number_input("재고유지비 /개/월", 0.0, 50.0, 2.0,  0.5)
        cost_short = st.number_input("부재고비 /개/월",   0.0, 50.0, 5.0,  0.5)
        cost_mat   = st.number_input("재료비 /개",        0,   500,  10,   1)
        cost_sub   = st.number_input("하청비 /개",        0,   500,  30,   1)

    with st.expander("🔧 작업 조건"):
        work_days = st.number_input("작업일수 (일/월)",          1, 31,  20, 1)
        work_hrs  = st.number_input("작업시간 (시간/일)",         1, 24,  8,  1)
        max_ot    = st.number_input("초과시간 제한 (hr/인/월)",   0, 100, 10, 1)
        std_time  = st.number_input("작업표준시간 (hr/개)",       1, 24,  4,  1)

    st.markdown("---")
    st.caption("강의록 기반 · Pyomo LP/IP 최적화\nHongik University Data Science")

params = dict(
    wage_reg=wage_reg, wage_ot=wage_ot,
    cost_hire=cost_hire, cost_fire=cost_fire,
    cost_hold=cost_hold, cost_short=cost_short,
    cost_mat=cost_mat, cost_sub=cost_sub,
    work_days=work_days, work_hrs=work_hrs,
    max_ot=max_ot, std_time=std_time,
    W0=W0, I0=I0, I_final=I_final, S0=0,
)

# session_state
for k in ["res_lp","res_ip","res_rnd"]:
    if k not in st.session_state:
        st.session_state[k] = None


# ══════════════════════════════════════════════════════════════
# 4. 메인 UI — 탭 기반
# ══════════════════════════════════════════════════════════════
st.markdown("# 🌿 원예장비 제조업체 총괄생산계획 최적화")
st.markdown(
    "**강의록 `setup_model` 구조 기반** | "
    "LP(이론 하한) → 반올림(실무 근사) → IP(정수 최적해) · Duality Gap 교육 도구"
)

TAB_NAMES = ["🚀 최적화 실행", "📊 생산·재고 대시보드",
             "👷 인력·잔업", "💰 비용 분석",
             "⚡ LP vs IP 비교", "🔬 민감도 분석", "📐 모델 수식"]
tabs = st.tabs(TAB_NAMES)


# ══════════════════════════════════════════════════════════
# TAB 0 — 최적화 실행
# ══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="step-header"><h3>STEP 1 · LP — 이론적 최적해</h3></div>',
                unsafe_allow_html=True)
    st.info("LP는 작업자·생산량을 **소수**로 허용해 계산 복잡도가 낮고 이론적 **하한(lower bound)**을 제공합니다.", icon="💡")

    if st.button("▶ LP 최적화 실행", type="primary", use_container_width=True, key="btn_lp"):
        with st.spinner("Pyomo LP 풀이 중 (GLPK Simplex)…"):
            res, status = run_model(D, params, "LP")
        if res is None:
            st.error(f"LP 풀이 실패: {status}")
        else:
            st.session_state.res_lp  = res
            st.session_state.res_rnd = None
            st.session_state.res_ip  = None
            st.success(f"✅ LP 최적해 도출 완료 — Minimal Cost = ₩{res['cost']:,.2f}천")

    if st.session_state.res_lp:
        res_lp = st.session_state.res_lp
        df_lp  = res_lp["df"]
        months = df_lp["월"].tolist()

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("최적 총비용", f"₩{res_lp['cost']:,.0f}천")
        c2.metric("총 수요",    f"{sum(D):,}개")
        c3.metric("총 생산",    f"{df_lp['P'].sum():,.0f}개")
        c4.metric("총 외주",    f"{df_lp['C'].sum():,.0f}개")
        c5.metric("최종 재고",  f"{df_lp['I'].iloc[-1]:,.0f}개")
        c6.metric("부족재고",   f"{df_lp['S'].sum():,.0f}개",
                  delta="없음 ✓" if df_lp["S"].sum()<0.01 else "발생 ⚠",
                  delta_color="normal" if df_lp["S"].sum()<0.01 else "inverse")

        with st.expander("📋 LP 강의록 형식 출력"):
            st.code(
                f"Minimal Cost = {res_lp['cost']}\n"
                f"(수요)   D = {[0]+D}\n"
                f"(작업자) W = {[W0]+[round(v,4) for v in df_lp['W'].tolist()]}\n"
                f"(고용)   H = {[None]+[round(v,4) for v in df_lp['H'].tolist()]}\n"
                f"(해고)   L = {[None]+[round(v,4) for v in df_lp['L'].tolist()]}\n"
                f"(생산)   P = {[None]+[round(v,4) for v in df_lp['P'].tolist()]}\n"
                f"(재고)   I = {[I0]+[round(v,4) for v in df_lp['I'].tolist()]}\n"
                f"(부재고) S = {[0]+[round(v,4) for v in df_lp['S'].tolist()]}\n"
                f"(외주)   C = {[None]+[round(v,4) for v in df_lp['C'].tolist()]}\n"
                f"(잔업)   O = {[None]+[round(v,4) for v in df_lp['O'].tolist()]}",
                language="text")

    st.divider()

    # ── STEP 2: Rounding ──────────────────────────────────
    st.markdown('<div class="step-header"><h3>STEP 2 · 반올림 시뮬레이터</h3></div>',
                unsafe_allow_html=True)

    if not st.session_state.res_lp:
        st.warning("먼저 LP를 실행하세요.")
    else:
        st.info("LP의 소수 작업자 수를 정수로 조정하면 **추가비용**과 **제약위반 여부**를 자동 검증합니다.", icon="🔄")
        df_lp = st.session_state.res_lp["df"]
        TH    = len(D)

        # 자동 버튼
        rb1, rb2, rb3 = st.columns([1,1,4])
        auto_ceil  = rb1.button("⬆ 전체 올림",   use_container_width=True)
        auto_round = rb2.button("↕ 전체 반올림", use_container_width=True)

        default_W = []
        for i in range(TH):
            lp_v = df_lp["W"].iloc[i]
            if auto_ceil:  default_W.append(math.ceil(lp_v))
            elif auto_round: default_W.append(round(lp_v))
            else: default_W.append(math.ceil(lp_v))

        cols_ = st.columns(TH)
        W_int = []
        for i,col_ in enumerate(cols_):
            with col_:
                st.markdown(f"**{i+1}월**")
                st.caption(f"LP: {df_lp['W'].iloc[i]:.2f}")
                w_ = st.number_input("", min_value=0, value=default_W[i],
                                     step=1, key=f"rnd_{i}",
                                     label_visibility="collapsed")
                W_int.append(w_)

        if st.button("▶ 반올림 시뮬레이션 실행", type="primary",
                     use_container_width=True, key="btn_rnd"):
            with st.spinner("고정된 W로 재LP 최적화 중…"):
                res_r, status = simulate_rounding(D, params, W_int)
            if res_r is None:
                st.error(f"반올림 시뮬레이션 실패 ({status}). 작업자 수를 늘려보세요.")
            else:
                st.session_state.res_rnd = res_r

        if st.session_state.res_rnd:
            res_r  = st.session_state.res_rnd
            res_lp = st.session_state.res_lp
            gap    = res_r["cost"] - res_lp["cost"]
            pct    = gap / res_lp["cost"] * 100
            short  = res_r["df"]["S"].max()

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("LP 하한",        f"₩{res_lp['cost']:,.0f}천")
            m2.metric("반올림 비용",     f"₩{res_r['cost']:,.0f}천",
                      delta=f"+₩{gap:,.0f}천 ({pct:+.2f}%)", delta_color="inverse")
            m3.metric("추가비용",        f"₩{gap:,.0f}천")
            m4.metric("부족재고 최대",   f"{short:,.1f}개",
                      delta="제약위반 ⚠" if short>0.01 else "실현가능 ✓",
                      delta_color="inverse" if short>0.01 else "normal")

            if short > 0.01:
                st.markdown(f'<div class="gap-box">⚠️ 일부 월에서 부족재고 {short:,.1f}개 발생 — 작업자 수를 늘리거나 외주를 활용하세요.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="ok-box">✅ 모든 제약조건 충족 — 반올림 계획이 실현 가능합니다.</div>',
                            unsafe_allow_html=True)

    st.divider()

    # ── STEP 3: IP ────────────────────────────────────────
    st.markdown('<div class="step-header"><h3>STEP 3 · IP — 정수 최적해 & Duality Gap</h3></div>',
                unsafe_allow_html=True)
    st.info("IP는 정수 제약으로 복잡도가 높지만 **진정한 정수 최적해**를 제공합니다. LP≤IP 관계로 Duality Gap을 정의합니다.", icon="🔢")

    if st.button("▶ IP 최적화 실행 (참고용)", use_container_width=True, key="btn_ip"):
        with st.spinner("Pyomo IP Branch & Bound 풀이 중… (수 초 소요)"):
            res_i, status = run_model(D, params, "IP")
        if res_i is None:
            st.error(f"IP 풀이 실패: {status}")
        else:
            st.session_state.res_ip = res_i
            st.success(f"✅ IP 최적해 도출 — Minimal Cost = ₩{res_i['cost']:,.0f}천")

    if st.session_state.res_ip and st.session_state.res_lp:
        res_ip = st.session_state.res_ip
        res_lp = st.session_state.res_lp
        res_rnd= st.session_state.res_rnd

        lp_c = res_lp["cost"]; ip_c = res_ip["cost"]
        gap  = ip_c - lp_c;   pct  = gap/lp_c*100

        g1,g2,g3 = st.columns(3)
        g1.metric("LP 하한",        f"₩{lp_c:,.0f}천")
        g2.metric("IP 정수 최적해", f"₩{ip_c:,.0f}천",
                  delta=f"+₩{gap:,.0f}천 ({pct:+.2f}%)", delta_color="inverse")
        g3.metric("Duality Gap",   f"₩{gap:,.0f}천 ({pct:.2f}%)",
                  delta="Gap 작음 ✓" if pct<1 else "Gap 주의 ⚠",
                  delta_color="normal" if pct<1 else "inverse")

        rnd_c = res_rnd["cost"] if res_rnd else ip_c
        wc1, wc2 = st.columns(2)
        wc1.plotly_chart(fig_gap_bar(lp_c, rnd_c, ip_c),  use_container_width=True)
        wc2.plotly_chart(fig_waterfall(lp_c, rnd_c, ip_c), use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 1 — 생산·재고 대시보드
# ══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### 📊 생산·재고 대시보드")
    available = {k: st.session_state[k] for k in ["res_lp","res_rnd","res_ip"]
                 if st.session_state[k]}
    if not available:
        st.info("TAB '🚀 최적화 실행'에서 먼저 모델을 실행하세요.", icon="ℹ️")
    else:
        results = list(available.values())
        st.plotly_chart(fig_production(results, D), use_container_width=True)
        st.plotly_chart(fig_inventory(results, I_final), use_container_width=True)

        # 월별 수급 상태 표
        st.markdown("#### 📋 월별 수급 상태 (LP 기준)")
        if st.session_state.res_lp:
            df_ = st.session_state.res_lp["df"].copy()
            df_["수급상태"] = df_.apply(
                lambda r: "⚠ 부족" if r["S"]>0.01 else ("📦 과잉" if r["I"]>500 else "✅ 적정"), axis=1)
            df_["공급계 (생산+외주)"] = df_["P"] + df_["C"]
            show_cols = ["월","수요","공급계 (생산+외주)","재고(I)","부족재고(S)","수급상태"]
            df_show   = df_.rename(columns={"I":"재고(I)","S":"부족재고(S)"})[
                ["월","수요","공급계 (생산+외주)","재고(I)","부족재고(S)","수급상태"]]
            st.dataframe(df_show.style.applymap(
                lambda v: "background-color:#fff5f5" if "⚠" in str(v) else
                          "background-color:#f5fff5" if "✅" in str(v) else "",
                subset=["수급상태"]), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — 인력·잔업
# ══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 👷 인력 계획 대시보드")
    available = {k: st.session_state[k] for k in ["res_lp","res_rnd","res_ip"]
                 if st.session_state[k]}
    if not available:
        st.info("먼저 모델을 실행하세요.", icon="ℹ️")
    else:
        results = list(available.values())
        st.plotly_chart(fig_workforce(results), use_container_width=True)
        st.plotly_chart(fig_overtime(results),  use_container_width=True)

        if st.session_state.res_lp:
            df_ = st.session_state.res_lp["df"]
            st.markdown("#### LP 작업자 수 소수 상세")
            fig_w = go.Figure()
            fig_w.add_bar(x=df_["월"], y=df_["W"],
                          text=[f"{v:.2f}명" for v in df_["W"]],
                          textposition="outside", marker_color="#4C78A8")
            fig_w.add_scatter(x=df_["월"],
                              y=[math.ceil(v) for v in df_["W"]],
                              mode="markers", marker=dict(color="#E45756",size=10,symbol="triangle-up"),
                              name="올림(ceil)")
            fig_w.update_layout(**LAYOUT, title="LP 작업자 수 (소수) vs 올림값",
                                yaxis_title="명")
            st.plotly_chart(fig_w, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 3 — 비용 분석
# ══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 💰 비용 분석 대시보드")
    available = {k: st.session_state[k] for k in ["res_lp","res_rnd","res_ip"]
                 if st.session_state[k]}
    if not available:
        st.info("먼저 모델을 실행하세요.", icon="ℹ️")
    else:
        sel = st.selectbox("분석 모델 선택", list(available.keys()),
                           format_func=lambda k: {"res_lp":"LP","res_rnd":"반올림","res_ip":"IP"}[k])
        res_sel = available[sel]
        cc1, cc2 = st.columns(2)
        cc1.plotly_chart(fig_cost_stack(res_sel), use_container_width=True)
        cc2.plotly_chart(fig_cost_pie(res_sel),   use_container_width=True)

        # 비용 유형별 합계 카드
        st.markdown("#### 비용 항목별 합계")
        df_ = res_sel["df"]
        card_cols = st.columns(4)
        for i,(key,lbl,clr) in enumerate(COST_META):
            total = df_[key].sum()
            pct_  = total/res_sel["cost"]*100
            card_cols[i%4].markdown(
                f'<div class="metric-card" style="border-left-color:{clr}">'
                f'<h4>{lbl}</h4><p>₩{total:,.0f}천</p>'
                f'<small style="color:#6c757d">{pct_:.1f}%</small></div>',
                unsafe_allow_html=True)

        # 상세 테이블
        with st.expander("📋 월별 상세 계획표"):
            rename_map = {"W":"작업자","H":"고용","L":"해고","P":"생산","I":"재고",
                          "S":"부족재고","C":"외주","O":"잔업(hr)","월비용":"총비용(천원)"}
            show = df_[["월","수요","W","H","L","P","I","S","C","O","월비용"]].rename(columns=rename_map)
            st.dataframe(show.style.format({c:"{:,.2f}" for c in show.select_dtypes("number").columns}),
                         use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# TAB 4 — LP vs IP 비교
# ══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### ⚡ LP · 반올림 · IP 종합 비교")
    available = {k: st.session_state[k] for k in ["res_lp","res_rnd","res_ip"]
                 if st.session_state[k]}
    if len(available) < 2:
        st.info("최적화 실행 탭에서 LP와 IP를 모두 실행해야 비교할 수 있습니다.", icon="ℹ️")
    else:
        results = list(available.values())
        # 종합 비교 테이블
        lp_c = st.session_state.res_lp["cost"] if st.session_state.res_lp else None
        rows_ = []
        for r in results:
            df_ = r["df"]
            gap_ = f"+₩{r['cost']-lp_c:,.0f}천 ({(r['cost']-lp_c)/lp_c*100:+.2f}%)" if lp_c and r["type"]!="LP" else "기준값"
            rows_.append({
                "모델": r["type"],
                "총비용 (천원)": f"₩{r['cost']:,.0f}",
                "LP 대비 Gap":  gap_,
                "총생산 (개)":  f"{df_['P'].sum():,.1f}",
                "총외주 (개)":  f"{df_['C'].sum():,.1f}",
                "최종재고 (개)":f"{df_['I'].iloc[-1]:,.1f}",
                "부족재고":     "⚠ 있음" if df_["S"].max()>0.01 else "✅ 없음",
                "정수해":       "✓" if r["type"] in ("IP","Rounding") else "✗ (소수)",
            })
        st.dataframe(pd.DataFrame(rows_), use_container_width=True, hide_index=True)

        t1,t2,t3 = st.tabs(["생산·수요","재고","작업자"])
        with t1: st.plotly_chart(fig_production(results,D),    use_container_width=True)
        with t2: st.plotly_chart(fig_inventory(results,I_final),use_container_width=True)
        with t3: st.plotly_chart(fig_workforce(results),        use_container_width=True)

        st.markdown("""
> **이론적 관계**
> $$Z^*_{\\text{LP}} \\;\\leq\\; Z^*_{\\text{Rounding}} \\;\\leq\\; Z^*_{\\text{IP}}$$
> LP는 정수 완화(relaxation)로 항상 하한 제공 · IP는 진정한 정수 최적해
""")


# ══════════════════════════════════════════════════════════
# TAB 5 — 민감도 분석
# ══════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### 🔬 파라미터 민감도 분석")
    st.info("특정 비용 파라미터를 변화(×0.5 ~ ×2.0)시켰을 때 LP 최적 총비용이 어떻게 바뀌는지 분석합니다.", icon="🔬")

    if not st.session_state.res_lp:
        st.warning("먼저 LP를 실행해야 기준값을 설정할 수 있습니다.")
    else:
        sens_options = {
            "정규임금 (wage_reg)":        "wage_reg",
            "초과임금 (wage_ot)":          "wage_ot",
            "고용비 (cost_hire)":          "cost_hire",
            "해고비 (cost_fire)":          "cost_fire",
            "재고유지비 (cost_hold)":       "cost_hold",
            "부재고비 (cost_short)":        "cost_short",
            "재료비 (cost_mat)":            "cost_mat",
            "하청비 (cost_sub)":            "cost_sub",
        }
        sel_label = st.selectbox("분석할 파라미터", list(sens_options.keys()))
        sel_key   = sens_options[sel_label]

        if st.button("🔬 민감도 분석 실행", type="primary", use_container_width=True):
            base_cost = st.session_state.res_lp["cost"]
            with st.spinner("민감도 분석 중 (6개 배율 × LP)…"):
                st.plotly_chart(fig_sensitivity(base_cost, sel_key, sel_label, D, params),
                                use_container_width=True)

        st.markdown("#### 다중 파라미터 영향도 (기준값 대비 ±50%)")
        if st.button("📊 전체 파라미터 영향도 계산", use_container_width=True):
            base_c = st.session_state.res_lp["cost"]
            impact_rows = []
            with st.spinner("파라미터별 영향도 계산 중…"):
                for lbl, key in sens_options.items():
                    for mult, tag in [(0.5,"-50%"),(1.5,"+50%")]:
                        p_tmp = {**params, key: params[key]*mult}
                        r_tmp, _ = run_model(D, p_tmp, "LP")
                        if r_tmp:
                            chg = (r_tmp["cost"]-base_c)/base_c*100
                            impact_rows.append({"파라미터":lbl,"변화":tag,
                                                "비용변화(%)":round(chg,2),
                                                "절대값(천원)":round(r_tmp["cost"]-base_c,0)})
            df_imp = pd.DataFrame(impact_rows)
            if not df_imp.empty:
                fig_imp = px.bar(df_imp, x="파라미터", y="비용변화(%)", color="변화",
                                 barmode="group", color_discrete_map={"-50%":"#4C78A8","+50%":"#E45756"},
                                 title="파라미터 ±50% 변화 시 비용 변화율 (%)")
                fig_imp.update_layout(**LAYOUT)
                st.plotly_chart(fig_imp, use_container_width=True)
                st.dataframe(df_imp.sort_values("비용변화(%)",ascending=False),
                             use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# TAB 6 — 모델 수식
# ══════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 📐 Pyomo 모델 수식 (강의록 기반)")
    st.markdown(r"""
#### 결정변수  ($t = 1 \ldots T$)

| 변수 | 의미 | 단위 |
|------|------|------|
| $W_t$ | t월 종업원 수 | 인/월 |
| $H_t$ | t월 신규 고용 수 | 인/월 |
| $L_t$ | t월 해고 수 | 인/월 |
| $P_t$ | t월 생산량 | 개/월 |
| $I_t$ | t월 말 재고 | 개/월 |
| $S_t$ | t월 말 부족재고 | 개/월 |
| $C_t$ | t월 하청 수량 | 개/월 |
| $O_t$ | t월 총 초과근무시간 | hr/월 |

#### 목적함수 — 총비용 최소화

$$Z = \sum_{t=1}^{T}\Bigl(
  \underbrace{640W_t}_{\text{정규노동}}
+ \underbrace{6O_t}_{\text{잔업}}
+ \underbrace{300H_t}_{\text{고용}}
+ \underbrace{500L_t}_{\text{해고}}
+ \underbrace{2I_t}_{\text{재고유지}}
+ \underbrace{5S_t}_{\text{부족재고}}
+ \underbrace{10P_t}_{\text{재료}}
+ \underbrace{30C_t}_{\text{하청}}
\Bigr)$$

#### 제약조건

| # | 명칭 | 수식 |
|---|------|------|
| ① | 노동력 균형 | $W_t = W_{t-1}+H_t-L_t$ |
| ② | 생산능력 | $P_t \leq 40W_t + \tfrac{1}{4}O_t$ |
| ③ | 재고균형 | $I_t = I_{t-1}+P_t+C_t-D_t-S_{t-1}+S_t$ |
| ④ | 초과근무 상한 | $O_t \leq 10W_t$ |
| ⑤ | 비음수 | $W_t,H_t,L_t,P_t,I_t,S_t,C_t,O_t \geq 0$ |
| ⑥ | 초기값 | $W_0=80,\;I_0=1000,\;S_0=0$ |
| ⑦ | 최종값 | $I_T \geq 500,\;S_T=0$ |

#### 생산능력 계수 도출

$$\text{규정시간 최대 생산} = \frac{1}{4}\,\frac{\text{ea}}{\text{hr}} \times 8\,\frac{\text{hr}}{\text{day}} \times 20\,\frac{\text{day}}{\text{mon}} = 40\;\frac{\text{ea}}{\text{인}\cdot\text{월}}$$

#### LP vs IP Relaxation 관계

$$Z^*_{\text{LP}} \;\leq\; Z^*_{\text{Rounding}} \;\leq\; Z^*_{\text{IP}}$$

- **LP**: 정수 제약 완화(relaxation) → 이론적 하한
- **반올림**: LP 결과를 정수화 → 실무 근사, 항상 실현 가능하지 않을 수 있음
- **IP**: Branch & Bound → 진정한 정수 최적해, 복잡도 $O(2^n)$

#### Duality Gap

$$\text{Duality Gap} = Z^*_{\text{IP}} - Z^*_{\text{LP}} \;\geq\; 0$$

Gap이 작을수록 LP 반올림이 좋은 근사치 → 실무에서 LP 우선 활용 근거
""")

    st.markdown("---")
    st.caption("강의록: Chunghun Ha, Hongik University | Data Science 핵심 > 08 스마트제조 > 총괄생산계획")
