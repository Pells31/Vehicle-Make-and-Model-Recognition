import streamlit as st
from predict import predict
import pandas as pd

if __name__ == '__main__':

    st.title('Vehicle Make, Model, and Year Recognition')

    file = st.file_uploader("Upload Your Image")

    if file:
        classname_matches = predict(file, model_name="resnet50_40epochs.pt", k=3)
        classname_matches = pd.DataFrame(data=classname_matches, columns=["Top 3 Predictions"])
        # classname_matches = classname_matches.set_axis(["Top 5 Predictions"], axis=1)

        st.image(file)

        st.write("The top 3 most likely vehicles are:")

        # CSS to inject contained in a string
        hide_table_row_index = """
                    <style>
                    thead tr th:first-child {display:none}
                    tbody th {display:none}
                    </style>
                    """

        hide_table_col_index = """
                    <style>
                    thead tr th:first-child {display:none}
                    tbody th {display:none}
                    </style>
                    """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        # Display a static table
        st.table(classname_matches)
