import streamlit as st
from nerif.agent import VisionAgent
from nerif.agent import MessageType
# from nerif.agent import EmbeddingModel
from PIL import Image
import base64
import io
import streamlit as st

# Set Streamlit theme to light mode
st.set_page_config(page_title="Figure Reader", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
# Set Streamlit theme to light mode
# Force light mode
st.markdown("""
    <style>
        .stApp {
            background-color: white;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)



def read_figure(agent, image, abstract, legend, additional_info):
    if agent.model == "openrouter/mistralai/pixtral-12b":
        agent.append_message(
            MessageType.TEXT,
            "This figure comes from a biology paper. With the information given, describe the figure in detail.",
        )
        if abstract != "":
            agent.append_message(MessageType.TEXT, "The paper is about (here comes the abstract): " + abstract + "\n")
    agent.append_message(MessageType.IMAGE_BASE64, image)
    if legend != "":
        agent.append_message(MessageType.TEXT, "Here is the legend(our figure's legend is contained in the following paragraph, just read the **RELATED** part): " + legend + "\n")
    if additional_info != "":
        agent.append_message(MessageType.TEXT, "Additional(This is hint for the description of the figure, sometimes it's just nothing): " + additional_info + "\n")
    if agent.model == "openrouter/mistralai/pixtral-12b":
        agent.append_message(
            MessageType.TEXT,
            "Try to Understand the figure first, then return the detailed description of the figure.",
        )
    else:
        agent.append_message(
            MessageType.TEXT,
            "Read the figure",
        )
    response = agent.chat()
    return response


def main():
    st.title("Figure Reader")

    openai_agent = VisionAgent(model="gpt-4o")
    llama_agent = VisionAgent(
        model="openrouter/meta-llama/llama-3.2-90b-vision-instruct"
    )
    pixtral_agent = VisionAgent(model="openrouter/mistralai/pixtral-12b")
    openai_agent.temperature = 0.30
    llama_agent.temperature = 0.30
    pixtral_agent.temperature = 0.65
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    input_abstract = st.text_input("Enter the abstract of the paper...")
    input_legend = st.text_input("Enter the legend of the figure...")
    input_additional_info = st.text_input(
        "Enter any additional info about the figure... For example, 'I care about the ratio of the bars in the figure, \
        it can tell me something about the significance of different genes'."
    )
    # input_prompt = st.text_input("Enter a prompt for the agent...")
    
    input_prompt = st.text_input("Enter a prompt for the agent. If you don't have any specific request, just leave it blank.")

    if st.button("Read Figure"):
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)
            # convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            # Read the image and display results with loading status
            with st.spinner("OpenAI Agent processing..."):
                result1 = read_figure(
                    openai_agent,
                    image_base64,
                    input_abstract,
                    input_legend,
                    input_additional_info,
                )
                with st.expander("OpenAI Agent Result", expanded=True):
                    st.markdown(
                        f'<div style="background-color: rgba(0, 100, 0, 0.75); color: #ffffff; padding: 10px; border-radius: 5px;">{result1}</div>',
                        unsafe_allow_html=True,
                    )

            with st.spinner("Llama Agent processing..."):
                result2 = read_figure(
                    llama_agent,
                    image_base64,
                    input_abstract,
                    input_legend,
                    input_additional_info,
                )
                with st.expander("Llama Agent Result", expanded=True):
                    st.markdown(
                        f'<div style="background-color: rgba(0, 100, 0, 0.75); color: #ffffff; padding: 10px; border-radius: 5px;">{result2}</div>',
                        unsafe_allow_html=True,
                    )

            with st.spinner("RYZE Agent processing..."):
                result3 = read_figure(
                    pixtral_agent,
                    image_base64,
                    input_abstract,
                    input_legend,
                    input_additional_info,
                )
                with st.expander("RYZE Agent Result", expanded=True):
                    st.markdown(
                        f'<div style="background-color: rgba(0, 100, 0, 0.75); color: #ffffff; padding: 10px; border-radius: 5px;">{result3}</div>',
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
