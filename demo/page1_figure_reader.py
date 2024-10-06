import streamlit as st
from nerif.agent import VisionAgent
from nerif.agent import MessageType
from PIL import Image
import base64
import io


def read_figure(agent, image, abstract, legend, additional_info):
    agent.append_message(MessageType.TEXT, "This figure comes from a biology paper. With the information given, describe the figure in detail.")
    agent.append_message(MessageType.TEXT, "Abstract: " + abstract + "\n")
    agent.append_message(MessageType.TEXT, "Legend: " + legend + "\n")
    agent.append_message(MessageType.TEXT, "Additional Info: " + additional_info + "\n")
    agent.append_message(
        MessageType.IMAGE_BASE64,
        image
    )
    response = agent.chat()
    return response

def main():
    st.title("Figure Reader")

    openai_agent = VisionAgent(model="gpt-4o")
    llama_agent = VisionAgent(model="openrouter/meta-llama/llama-3.2-90b-vision-instruct")
    pixtral_agent = VisionAgent(model="openrouter/mistralai/pixtral-12b")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    input_abstract = st.text_input("Enter the abstract of the paper...")
    input_legend = st.text_input("Enter the legend of the figure...")
    input_additional_info = st.text_input("Enter any additional info about the figure...")
    # input_prompt = st.text_input("Enter a prompt for the agent...")
    
    if st.button("Read Figure"):
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            # convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            # Read the image and display results with loading status
            with st.spinner("OpenAI Agent processing..."):
                result1 = read_figure(openai_agent, image_base64, input_abstract, input_legend, input_additional_info)
                with st.expander("OpenAI Agent Result", expanded=True):
                    st.markdown(
                        f'<div style="background-color: #e6ffe6; padding: 10px; border-radius: 5px;">{result1}</div>',
                        unsafe_allow_html=True
                    )

            with st.spinner("Llama Agent processing..."):
                result2 = read_figure(llama_agent, image_base64, input_abstract, input_legend, input_additional_info)
                with st.expander("Llama Agent Result", expanded=True):
                    st.markdown(
                        f'<div style="background-color: #e6ffe6; padding: 10px; border-radius: 5px;">{result2}</div>',
                        unsafe_allow_html=True
                    )

            with st.spinner("Pixtral Agent processing..."):
                result3 = read_figure(pixtral_agent, image_base64, input_abstract, input_legend, input_additional_info)
                with st.expander("Pixtral Agent Result", expanded=True):
                    st.markdown(
                        f'<div style="background-color: #e6ffe6; padding: 10px; border-radius: 5px;">{result3}</div>',
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()