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

class Intro(TeethApp):
    def __init__(self):
        TeethApp.__init__(self)
        self.build_app()

    def build_app(self):
        st.title("AI-assited Tooth Segmentation")
        st.markdown("This app automatically segments intra-oral scans of teeth using machine learning.")
        st.markdown("Head to the 'Segment' tab to try it out!")
        st.markdown("**Example:**")
        st.image("illustration.png")

if __name__ == "__main__":
    app = Intro()