import streamlit as st
from transformers import pipeline

# Set up the Streamlit app
st.title("Language Learning Model (LLM) Demo")
st.write("This is a simple application using a pre-trained language model to demonstrate text generation.")

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = pipeline("text-generation", model="gpt-2")
    return model

model = load_model()

# User input
prompt = st.text_area("Enter a prompt:", "Once upon a time")

# Generate text
if st.button("Generate"):
    with st.spinner("Generating text..."):
        generated_text = model(prompt, max_length=50, num_return_sequences=1)
        st.success("Text generated successfully!")
        st.write(generated_text[0]["generated_text"])

# Footer
st.markdown("---")
st.write("Developed by Prashant Shukla(https://https://github.com/Prashant2091)")
