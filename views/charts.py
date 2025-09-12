import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def render_bar_chart(df, x, y, color, title, category_order):
    fig = px.bar(df, x=x, y=y, color=color, barmode="relative", title=title, height=400)
    fig.update_layout(width=1000, bargap=0.2)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=category_order)
    st.plotly_chart(fig, use_container_width=False)

def render_line_chart(df, x, y, color, title):
    fig = px.line(df, x=x, y=y, color=color, markers=True, title=title, height=400)
    fig.update_layout(width=1000, legend_title_text=color)
    st.plotly_chart(fig, use_container_width=False)

def render_grouped_bar_and_line(prod_df, demand_df, x_col, y_col, category_col, title):
    fig = go.Figure()

    # Bar traces
    for cat in prod_df[category_col].unique():
        sub = prod_df[prod_df[category_col] == cat]
        fig.add_trace(go.Bar(x=sub[x_col], y=sub[y_col], name=cat))

    # Line traces
    for cat in demand_df[category_col].unique():
        sub = demand_df[demand_df[category_col] == cat]
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=sub[y_col],
            name=f"{cat} (scenario)",
            mode="lines+markers"
        ))

    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Year",
        yaxis_title="ktoe",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
