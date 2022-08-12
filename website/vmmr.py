import streamlit as st


if __name__ == '__main__':

    st.title('Vehicle Make, Model, and Year Recognition')

    st.write("Upload your vehicle image for classification!")

    file = st.file_uploader("Upload Your Image")

    img = Image.open(file)