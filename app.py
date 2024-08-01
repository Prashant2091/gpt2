import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Set up the Streamlit app
st.title("Language Learning Model (LLM) Demo")
st.write("This is a simple application using a pre-trained language model to demonstrate text generation.")

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model_name = "distilgpt2"  # Using a smaller model for deployment ease
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return generator
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        return None

generator = load_model()

# User input
prompt = st.text_area("Enter a prompt:", "Once upon a time")

# Generate text
if st.button("Generate"):
    if generator:
        with st.spinner("Generating text..."):
            try:
                generated_text = generator(prompt, max_length=50, num_return_sequences=1)
                st.success("Text generated successfully!")
                st.write(generated_text[0]["generated_text"])
            except Exception as e:
                st.error(f"An error occurred during text generation: {str(e)}")
    else:
        st.error("Model could not be loaded. Please check the setup.")

# Footer
st.markdown("---")
st.write("Developed by [Prashant Shukla](https://github.com/Prashant2091)")
