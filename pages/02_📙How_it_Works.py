import streamlit as st
from streamlit import session_state as session

from PIL import Image

class TeethApp:
    def __init__(self):
        # Font
        with open("utils/style.css") as css:
            st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
    
        # Logo
        self.image_path = "utils/teeth-295404_1280.png"
        self.image = Image.open(self.image_path)
        width, height = self.image.size
        scale = 12
        new_width, new_height = width / scale, height / scale
        self.image = self.image.resize((int(new_width), int(new_height)))

        # Streamlit side navigation bar
        st.sidebar.markdown("# AI ToothSeg")
        st.sidebar.markdown("Automatic teeth segmentation with Deep Learning")
        st.sidebar.markdown(" ")
        st.sidebar.image(self.image, use_column_width=False)
        st.markdown(
            """
                <style>
                .css-1bxukto {
                background-color: rgb(255, 255, 255) ;""",
            unsafe_allow_html=True,
        )
        
# Configure Streamlit page
st.set_page_config(page_title="Teeth Segmentation", page_icon="ⓘ")

class Guide(TeethApp):
    def __init__(self):
        TeethApp.__init__(self)
        self.build_app()

    def build_app(self):
        st.title("More Coming Soon")
        st.markdown("Made by Huayuan Song for the 10 ECTS 02830 Advanced Project in Digital Media Technology course at the Technical University of Denmark.")
        st.markdown("ML backend is based on MeshSegNet architecture by [Lian et al.](https://ieeexplore.ieee.org/abstract/document/8984309).")
        st.markdown("The model has been trained on intra-oral scans of both upper and lower jaws annotated, validated by professionals in the 3DTeethSeg'22 Challenge by [Ben-Hamadou et al.](https://arxiv.org/abs/2305.18277).")
        st.markdown("**Thanks for trying the app out!**")
        st.image("illu.png")

if __name__ == "__main__":
    app = Guide()