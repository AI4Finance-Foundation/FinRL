import streamlit as st

def set_block_container_width(max_width: int = 1200):
    max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{
                {max_width_str}
            }}
        </style>
        """,
        unsafe_allow_html=True)
