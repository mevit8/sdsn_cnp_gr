import plotly.express as px
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