import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
            body {
                background-color: #0e1117;
                color: white;
            }

            .main-header {
                text-align: center;
                padding: 20px;
            }

            .card {
                border-radius: 12px;
                padding: 15px;
                background-color: #1f2933;
            }
        </style>
    """, unsafe_allow_html=True)
