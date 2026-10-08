from __future__ import annotations

from math import comb, sqrt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="EventLab | Probability Lab",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #071b3b;
        --cyan: #13b8c8;
        --ink: #102543;
        --line: #e4ebf4;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 78% 0%, rgba(19,184,200,.10), transparent 32%),
            linear-gradient(180deg, #f6f9fd 0%, #eef3f9 100%);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background:transparent !important; box-shadow:none !important; }
    .block-container { max-width:1440px; padding-top:4.5rem; padding-bottom:3rem; }
    .hero {
        position:relative; overflow:hidden; isolation:isolate;
        padding:1.35rem 1.7rem; border-radius:18px; margin-bottom:.9rem;
        color:white; background:linear-gradient(112deg,#071b3b 0%,#103969 60%,#087c91 130%);
        box-shadow:0 8px 24px rgba(7,27,59,.12);
    }
    .hero::before, .hero::after { display:none; }
    .hero-topline { display:flex; align-items:center; gap:9px; color:#aeeaf0; font-size:.74rem; letter-spacing:.16em; text-transform:uppercase; font-weight:700; }
    .signal-dot { width:8px; height:8px; border-radius:50%; background:#41e1d4; }
    .hero h1 { color:#fff; font-size:clamp(2rem,3vw,2.8rem); line-height:1.05; letter-spacing:-.035em; margin:.4rem 0; }
    .hero p { color:#e5eef8; max-width:900px; font-size:1rem; line-height:1.5; margin:0; }
    .hero-meta { color:#c2d4e8; font-size:.76rem; margin-top:.7rem; letter-spacing:.035em; }
    div[data-testid="stMetric"] {
        background:#fff; border:1px solid var(--line); border-radius:16px; padding:15px 17px;
        box-shadow:0 4px 14px rgba(17,43,78,.04);
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] p {
        color:#405670 !important; font-weight:650 !important; opacity:1 !important;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] div {
        color:#0b2851 !important; opacity:1 !important;
    }
    [data-testid="stSlider"] [data-testid="stWidgetLabel"],
    [data-testid="stSlider"] [data-testid="stWidgetLabel"] *,
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] label * {
        color:#0b2851 !important; opacity:1 !important; font-weight:650 !important;
    }
    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] * {
        color:#38536f !important; opacity:1 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.38rem; border-bottom:1px solid #dce5f0; }
    [data-testid="stTabs"] button[data-baseweb="tab"],
    [data-testid="stTabs"] [role="tab"],
    .stTabs button[data-baseweb="tab"] {
        background:#e8eef6 !important; border-radius:10px 10px 0 0; padding:.65rem 1rem;
        color:#38536f !important; font-weight:650 !important; opacity:1 !important;
    }
    [data-testid="stTabs"] button[data-baseweb="tab"] *,
    [data-testid="stTabs"] [role="tab"] *,
    .stTabs button[data-baseweb="tab"] * {
        color:#38536f !important; opacity:1 !important;
    }
    [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background:#fff !important; color:#0b2851 !important; opacity:1 !important;
    }
    [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] *,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] * {
        color:#0b2851 !important; opacity:1 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:#13b8c8 !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top:1.15rem; }
    h2, h3 { color:#0d2a50; letter-spacing:-.02em; }
    div[data-testid="stAlert"] { border-radius:14px; }
    div[data-testid="stDataFrame"] { border:1px solid #e4ebf4; border-radius:14px; overflow:hidden; }
    .small-caps { color:#66809b; font-size:.72rem; font-weight:750; letter-spacing:.14em; text-transform:uppercase; }
    .formula-card {
        min-height:100px; padding:1rem 1.15rem; margin:.25rem 0 .75rem;
        border:1px solid #e2eaf4; border-radius:16px; background:#fff;
        box-shadow:0 5px 18px rgba(17,43,78,.035);
    }
    .formula-card h3 { font-size:1rem; margin:0 0 .35rem; color:#163a63; }
    .formula-card p { color:#657991; margin:0; font-size:.9rem; line-height:1.5; }
    .footnote { color:#6c7e94; font-size:.82rem; line-height:1.55; }
    @media (max-width:700px) {
        .hero { padding:1.1rem; border-radius:16px; }
        .hero p { font-size:.94rem; }
        .block-container { padding-left:1rem; padding-right:1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def binomial_probability(total_nodes: int, overloaded_nodes: int, probability: float) -> float:
    """Probability that exactly overloaded_nodes out of total_nodes overload."""
    return comb(total_nodes, overloaded_nodes) * probability**overloaded_nodes * (1 - probability) ** (
        total_nodes - overloaded_nodes
    )


SIMULATION_TRIALS = 20_000


def chart_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=45, b=15),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#304764"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="#0b2851", font_color="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#dce5f0", zeroline=False)
    fig.update_yaxes(gridcolor="#e9eef5", zeroline=False)
    return fig


st.markdown(
    """
    <div class="hero">
      <div class="hero-topline"><span class="signal-dot"></span> Interactive probability model</div>
      <h1>EventLab</h1>
      <p>Explore how probability describes server overloads. Set the model, calculate exact outcomes, and compare the mathematics with simulated data.</p>
      <div class="hero-meta">INDIVIDUAL PROJECT &nbsp;·&nbsp; MAKHMETOV ALIKHAN &nbsp;·&nbsp; PROBABILITY IN ACTION</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("Set the model")
    control_n, control_p, control_k = st.columns(3)
    with control_n:
        node_count = st.slider(
            "Number of servers (n)", min_value=2, max_value=20, value=8, step=1,
            help="n is the total number of servers in the cluster.",
        )
        st.caption("n = the total number of servers in the cluster.")
    with control_p:
        overload_rate = st.slider(
            "Overload chance per server (p)", min_value=0.01, max_value=0.99, value=0.12, step=0.01, format="%.2f",
            help="p is the chance that one server overloads during one hypothetical minute. For example, 0.12 means 12%.",
        )
        st.caption("p = the probability one server overloads in one hypothetical minute; 0.12 means 12%.")
    with control_k:
        exact_count = st.slider(
            "Exact overloaded servers (k)", min_value=0, max_value=node_count, value=min(2, node_count),
            help="k is the exact number of overloaded servers in the outcome being studied.",
        )
        st.caption("k = the exact number of overloaded servers in the outcome being studied.")
    st.caption("These settings update every section. The simulation compares the model with 20,000 generated one-minute outcomes.")

p_column, assumptions_column = st.columns([1.2, 0.8])
with p_column:
    with st.container(border=True):
        st.markdown("### How can we get p?")
        st.write(
            "In this educational model, p is an assumed probability. In a real system, "
            "p can be estimated from historical server data. An observation means one server "
            "recorded over one time interval."
        )
        st.latex(r"p = \frac{\text{number of overloads}}{\text{number of observations}}")
        st.write("For example: 120 overloads / 1,000 observations = 0.12 = 12%.")
with assumptions_column:
    with st.container(border=True):
        st.markdown("### Model assumptions")
        st.markdown(
            "- All servers have the same overload probability.\n"
            "- Server results are independent.\n"
            "- Each trial is independent.\n"
            "- This is an educational model, not real server monitoring."
        )

tab_combinatorics, tab_simulation, tab_events, tab_formulas = st.tabs(
    ["Combinatorics", "Simulation", "Events", "Formulas"]
)


with tab_combinatorics:
    st.markdown('<div class="small-caps">Pattern space</div>', unsafe_allow_html=True)
    st.subheader("How many ways can a cluster overload?")
    st.write(
        "One outcome describes one hypothetical minute. Each server either stays safe or overloads. "
        "This section counts the different server combinations that can produce an exact overload count."
    )
    st.info(
        f"Symbols: n = {node_count} total servers; p = {overload_rate:.0%} chance one server overloads in one minute; "
        f"k = {exact_count} overloaded servers in the outcome. The computer evaluates each server independently."
    )

    exact_probability = binomial_probability(node_count, exact_count, overload_rate)
    single_pattern_probability = overload_rate**exact_count * (1 - overload_rate) ** (node_count - exact_count)
    all_nodes_probability = overload_rate**node_count
    pattern_count = comb(node_count, exact_count)

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric(f"Ways to choose exactly {exact_count} servers", f"{pattern_count:,}")
    metric_b.metric(f"Chance exactly {exact_count} overload", f"{exact_probability:.2%}")
    metric_c.metric("Chance all servers overload", f"{all_nodes_probability:.6%}")

    st.markdown("#### How the formula works")
    st.write(
        "First, choose which k servers overload: there are C(n, k) possible groups. "
        "For one specific group, k servers must overload and the other n − k must stay safe. "
        "Multiply those two parts, then multiply by the number of possible groups."
    )
    st.latex(r"\Pr(X=k)=\binom{n}{k}p^k(1-p)^{n-k}")
    st.caption(
        f"Here X is the number overloaded in one minute. With these settings, {pattern_count:,} possible groups × "
        f"{single_pattern_probability:.2%} chance for one specific group = {exact_probability:.2%} chance of exactly k = {exact_count}."
    )

    outcomes = list(range(node_count + 1))
    probabilities = [binomial_probability(node_count, r, overload_rate) for r in outcomes]
    ways = [comb(node_count, r) for r in outcomes]
    probability_colors = ["#13b8c8" if r == exact_count else "#7da9d2" for r in outcomes]
    pattern_colors = ["#13b8c8" if r == exact_count else "#9bb3ce" for r in outcomes]

    st.markdown("#### Read the charts")
    st.caption(
        "The left chart shows the model probability for each possible overload count. "
        "The right chart shows how many different server groups produce each count. "
        "The cyan bar marks the selected k."
    )
    left_chart, right_chart = st.columns(2)
    with left_chart:
        probability_fig = go.Figure(
            go.Bar(
                x=outcomes,
                y=probabilities,
                marker_color=probability_colors,
                hovertemplate="Overloaded servers: %{x}<br>Probability: %{y:.2%}<extra></extra>",
            )
        )
        probability_fig.update_layout(title="Chance of each outcome", xaxis_title="Overloaded servers", yaxis_title="Model probability")
        probability_fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(chart_layout(probability_fig), use_container_width=True, config={"displayModeBar": False})
    with right_chart:
        patterns_fig = go.Figure(
            go.Bar(
                x=outcomes,
                y=ways,
                marker_color=pattern_colors,
                hovertemplate="Overloaded servers: %{x}<br>Different server groups: %{y:,}<extra></extra>",
            )
        )
        patterns_fig.update_layout(title="Ways to form each outcome", xaxis_title="Overloaded servers", yaxis_title="Number of server groups")
        st.plotly_chart(chart_layout(patterns_fig), use_container_width=True, config={"displayModeBar": False})

    distribution = pd.DataFrame(
        {
            "Overloaded servers (r)": outcomes,
            "Different server groups C(n, r)": ways,
            "Chance of this outcome P(X = r)": probabilities,
        }
    )
    st.markdown("#### Outcome table")
    st.caption("Each row is one possible number of overloaded servers. The group count explains how many patterns make that row possible; the chance column gives its theoretical probability.")
    st.dataframe(
        distribution.style.format({"Chance of this outcome P(X = r)": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        '<div class="footnote">A full-cluster overload is the extreme-case outage: every server node overloads at once.</div>',
        unsafe_allow_html=True,
    )


with tab_simulation:
    st.markdown('<div class="small-caps">Prediction → simulated data → comparison</div>', unsafe_allow_html=True)
    st.subheader("Let the model play out")
    mean_theory = node_count * overload_rate
    sd_theory = sqrt(node_count * overload_rate * (1 - overload_rate))
    st.info(
        f"Prediction: the average will be near {mean_theory:.2f} overloaded servers per simulated minute "
        f"because the expected count is n × p = {node_count} × {overload_rate:.2f}."
    )

    st.markdown("#### What is one simulation trial?")
    st.write(
        f"The simulation uses {SIMULATION_TRIALS:,} computer-generated trials. Each trial represents one hypothetical one-minute snapshot: "
        "the program independently decides whether each server overloads, then counts the overloaded servers. "
        "These are not real elapsed minutes, real traffic, or measurements from a live server."
    )
    diagram_one, arrow_one, diagram_two, arrow_two, diagram_three = st.columns([1, 0.12, 1, 0.12, 1])
    with diagram_one:
        st.markdown('<div class="formula-card"><h3>1 · Set up one trial</h3><p>n servers; each has overload chance p.</p></div>', unsafe_allow_html=True)
    with arrow_one:
        st.markdown("### →")
    with diagram_two:
        st.markdown('<div class="formula-card"><h3>2 · Count X</h3><p>X is the number of servers overloaded in that trial.</p></div>', unsafe_allow_html=True)
    with arrow_two:
        st.markdown("### →")
    with diagram_three:
        st.markdown(f'<div class="formula-card"><h3>3 · Repeat 20,000 times</h3><p>Compare the observed outcomes with the formula.</p></div>', unsafe_allow_html=True)

    refresh_simulation = st.button("Generate a new random sample", type="primary")
    simulation_key = (node_count, round(overload_rate, 4))
    if (
        refresh_simulation
        or st.session_state.get("simulation_key") != simulation_key
        or "simulation_samples" not in st.session_state
    ):
        rng = np.random.default_rng()
        st.session_state["simulation_samples"] = rng.binomial(node_count, overload_rate, size=SIMULATION_TRIALS)
        st.session_state["simulation_key"] = simulation_key

    samples = st.session_state["simulation_samples"]
    observed_mean = float(np.mean(samples))
    observed_median = float(np.median(samples))
    observed_sd = float(np.std(samples))
    metric_mean, metric_median, metric_sd = st.columns(3)
    metric_mean.metric("Average overloaded servers", f"{observed_mean:.2f}", f"{observed_mean - mean_theory:+.2f} vs predicted")
    metric_median.metric("Middle observed outcome", f"{observed_median:g}", "servers overloaded")
    metric_sd.metric("Typical spread from average", f"{observed_sd:.2f}", f"{observed_sd - sd_theory:+.2f} vs predicted")
    st.caption("The mean is the average count per trial. The median is the middle count after sorting all trials. Standard deviation describes how spread out the counts are.")

    observed_counts = np.bincount(samples, minlength=node_count + 1)
    observed_shares = observed_counts / SIMULATION_TRIALS
    model_shares = [binomial_probability(node_count, r, overload_rate) for r in outcomes]
    selected_theory = model_shares[exact_count]
    selected_simulation = observed_shares[exact_count]
    selected_difference_pp = (selected_simulation - selected_theory) * 100

    with st.container(border=True):
        st.markdown("#### Selected outcome")
        st.write(f"Exactly **{exact_count}** servers overloaded")
        selected_theory_col, selected_simulation_col, selected_difference_col = st.columns(3)
        selected_theory_col.metric("Theory", f"{selected_theory:.2%}")
        selected_simulation_col.metric("Simulation", f"{selected_simulation:.2%}")
        selected_difference_col.metric(
            "Difference (simulation − theory)",
            f"{selected_difference_pp:+.2f} percentage points",
        )
        st.caption(
            f"Simulation is the share of {SIMULATION_TRIALS:,} trials with exactly k = {exact_count} overloaded servers. "
            "The difference is simulation minus theory."
        )

    comparison_fig = go.Figure()
    comparison_fig.add_trace(
        go.Bar(
            name="Simulation",
            x=outcomes,
            y=observed_shares,
            marker_color="#13b8c8",
            opacity=0.82,
            hovertemplate="Servers overloaded: %{x}<br>Fraction of trials: %{y:.2%}<extra></extra>",
        )
    )
    comparison_fig.add_trace(
        go.Scatter(
            name="Theory",
            x=outcomes,
            y=model_shares,
            mode="lines+markers",
            line=dict(color="#123d70", width=3),
            marker=dict(size=7, color="#123d70"),
            hovertemplate="Servers overloaded: %{x}<br>Formula predicts: %{y:.2%}<extra></extra>",
        )
    )
    comparison_fig.update_layout(title="What happened in the simulation vs. what theory predicts", barmode="overlay", xaxis_title="Servers overloaded in one trial", yaxis_title="Share of all trials")
    comparison_fig.update_yaxes(tickformat=".0%")
    st.caption("Cyan bars show the fraction of simulated trials with each result. The navy line shows the formula's predicted probability for that same result.")
    st.plotly_chart(chart_layout(comparison_fig, height=410), use_container_width=True, config={"displayModeBar": False})

    observed_percent = observed_shares * 100
    theoretical_percent = np.array(model_shares) * 100
    comparison = pd.DataFrame(
        {
            "Servers overloaded in one trial": outcomes,
            "Observed trials": observed_counts,
            "Observed share": observed_percent,
            "Theoretical share": theoretical_percent,
            "Gap (percentage points)": observed_percent - theoretical_percent,
        }
    )
    st.markdown("#### Frequency table")
    st.caption(f"Each row is an overload count. ‘Observed trials’ counts how often it occurred in the {SIMULATION_TRIALS:,} computer-generated trials. Shares are percentages; the gap is observed minus theoretical, measured in percentage points.")
    st.dataframe(
        comparison.style.format(
            {
                "Observed share": "{:.2f}%",
                "Theoretical share": "{:.2f}%",
                "Gap (percentage points)": "{:+.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Interpretation: observed bars will not match the theory perfectly in a finite run. "
        "With more trials, the observed frequencies usually move closer to the theoretical probabilities."
    )


with tab_events:
    st.markdown('<div class="small-caps">Conditional probability & independence</div>', unsafe_allow_html=True)
    st.subheader("How does new information change a probability?")
    at_least_one_probability = 1 - (1 - overload_rate) ** node_count
    selected_probability = binomial_probability(node_count, exact_count, overload_rate)
    conditional_probability = (
        selected_probability / at_least_one_probability if exact_count >= 1 and at_least_one_probability > 0 else 0.0
    )
    two_minute_probability = at_least_one_probability**2

    event_left, event_right = st.columns(2)
    with event_left:
        st.markdown("### Given that at least one server overloaded")
        st.write(
            f"Let A mean “one or more servers overload,” and B mean “exactly k = {exact_count} servers overload.” "
            "Conditional probability asks how likely B is after we already know A happened. When k is at least 1, B is part of A, so divide the chance of B by the chance of A."
        )
        st.latex(r"\Pr(A)=1-(1-p)^n")
        st.latex(r"\Pr(B\mid A)=\frac{\Pr(B)}{\Pr(A)}\quad (k\geq1)")
        st.metric("Chance at least one server overloads", f"{at_least_one_probability:.2%}")
        st.metric(f"Chance exactly {exact_count}, given at least one", f"{conditional_probability:.2%}")
        if exact_count == 0:
            st.caption("If k = 0, the events cannot happen together: A says at least one server overloaded. Therefore this conditional probability is 0.")
    with event_right:
        st.markdown("### Overloads in two separate minutes")
        st.write(
            "Assume one minute does not affect the next. Then the chance of an overload in minute 2 stays the same, even if minute 1 had an overload. "
            "This is the independence assumption."
        )
        st.latex(r"\Pr(A_1\cap A_2)=\Pr(A_1)\Pr(A_2)")
        st.metric("Chance both minutes have an overload", f"{two_minute_probability:.2%}")
        st.metric("Chance minute 2 overloads after minute 1", f"{at_least_one_probability:.2%}")

    st.markdown("---")
    st.markdown("#### Independence in one line")
    st.latex(r"\Pr(E\cap F)=\Pr(E)\Pr(F)\quad\text{when }E\text{ and }F\text{ are independent}")
    st.caption(
        "The two-minute result depends on the independence assumption. In a real system, one bad minute may overload queues and make the next minute riskier."
    )


with tab_formulas:
    st.markdown('<div class="small-caps">Defense-ready reference</div>', unsafe_allow_html=True)
    st.subheader("Formula deck")
    st.write("Each formula is followed by what it counts or predicts in this server example.")

    formula_rows = [
        (
            "Combinations",
            r"\binom{n}{k}=\frac{n!}{k!(n-k)!}",
            "Counts how many different groups of k servers can be selected from n servers. Order does not matter.",
        ),
        (
            "Binomial probability",
            r"\Pr(X=k)=\binom{n}{k}p^k(1-p)^{n-k}",
            "Chance exactly k servers overload: count the possible groups, then multiply by the chance of one such group.",
        ),
        (
            "At least one overload",
            r"\Pr(X\geq1)=1-(1-p)^n",
            "Use the complement: subtract the chance (1 − p)^n that every server stays safe.",
        ),
        (
            "Expected count and spread",
            r"\mathbb{E}[X]=np,\qquad \operatorname{Var}(X)=np(1-p),\qquad \sigma=\sqrt{np(1-p)}",
            "np is the expected average count per trial; σ describes the typical spread around that average.",
        ),
        (
            "Conditional probability",
            r"\Pr(B\mid A)=\frac{\Pr(A\cap B)}{\Pr(A)}",
            "Updates a probability after new information A is known.",
        ),
        (
            "Independent events",
            r"\Pr(A\cap B)=\Pr(A)\Pr(B)",
            "The probability that both events occur when neither changes the chance of the other.",
        ),
    ]

    for offset in range(0, len(formula_rows), 2):
        cols = st.columns(2)
        for col, (title, equation, explanation) in zip(cols, formula_rows[offset : offset + 2]):
            with col:
                st.markdown(f'<div class="formula-card"><h3>{title}</h3><p>{explanation}</p></div>', unsafe_allow_html=True)
                st.latex(equation)

    st.markdown("---")
    st.markdown("#### Symbol key")
    st.caption("These symbols appear in the formulas and controls. The current value column reflects the sliders.")
    symbol_table = pd.DataFrame(
        [
            ("n", str(node_count), "Total number of servers in the cluster"),
            ("p", f"{overload_rate:.2f} = {overload_rate:.0%}", "Chance one server overloads during one hypothetical minute"),
            ("k", str(exact_count), "Exact number of overloaded servers in the outcome being studied"),
            ("X", "0, 1, …, n", "Count of overloaded servers in one trial"),
            ("A", "X ≥ 1", "Event that one or more servers overload"),
        ],
        columns=["Symbol", "Current value", "Meaning"],
    )
    st.dataframe(symbol_table, use_container_width=True, hide_index=True)
    st.markdown(
        '<div class="footnote">Assumptions: every server uses the same p, server outcomes are independent within a trial, and trials are independent. '
        "Real servers can affect one another, so the model is a simplified teaching example.</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown(
    '<div class="footnote">EventLab · Theory of Probability · Individual project by Makhmetov Alikhan</div>',
    unsafe_allow_html=True,
)
