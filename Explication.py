import streamlit as st

def read_markdown_file(markdown_file):
    with open(markdown_file, 'r', encoding='utf-8') as file:
        return file.read()

st.set_page_config(layout="wide")

text_selected = st.selectbox(
    "Version",
    ['Résumée', 'Developpée']
)

if text_selected == 'Developpée':
    markdown_content = read_markdown_file("/Users/julesmourgues/Documents/Programmation/Actuel/Allocation/WhalesOptimizer-main/developped.md")
    st.markdown(markdown_content)  
else:
    markdown_content = read_markdown_file("/Users/julesmourgues/Documents/Programmation/Actuel/Allocation/WhalesOptimizer-main/summary.md")
    st.markdown(markdown_content)
