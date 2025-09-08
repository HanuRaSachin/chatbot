import streamlit as st
import google.generativeai as genai
from googletrans import Translator
import warnings

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Suppress warnings for a cleaner interface
warnings.filterwarnings("ignore")

# --- Google Gemini API Configuration ---
try:
    # Use Streamlit's secrets management for the API key
    GEMINI_API_KEY = st.secrets[AIzaSyDbvA_b9nXFEnSva2Q3pGW2H_H6ZpF8KD8]
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except (KeyError, Exception):
    st.error("🔴 **Error:** Google API Key not found. Please add it to your Streamlit secrets.", icon="🚨")
    st.stop()


# --- Translator Initialization ---
@st.cache_resource
def get_translator():
    return Translator()

translator = get_translator()


# --- Core Functions ---
def get_ai_response(prompt, language="en"):
    """
    Generates a response from the Gemini model and translates it.
    Includes a crucial safety disclaimer for health-related queries.
    """
    # Safety disclaimer to prepend to every health-related AI response
    disclaimer = (
        "**Disclaimer:** I am an AI assistant, not a medical professional. "
        "This information is for informational purposes only and should not be considered medical advice. "
        "Please consult with a qualified healthcare provider for any health concerns or before making any medical decisions."
    )
    try:
        full_prompt = (
            "You are a helpful AI health assistant. Provide informative and safe advice. "
            f"Here is the user's query: '{prompt}'"
        )
        response = model.generate_content(full_prompt)
        
        # Combine the original response with the disclaimer
        final_response_text = f"{response.text}\n\n---\n\n{disclaimer}"

        if language != "en":
            return translate_text(final_response_text, language)
        return final_response_text

    except Exception as e:
        return f"Error: Could not get a response from the AI model. Details: {str(e)}"

def translate_text(text, dest_language="en"):
    """Translates text to the specified destination language."""
    try:
        if text and text.strip():
            return translator.translate(text, dest=dest_language).text
        return ""
    except Exception as e:
        return f"Translation Error: {str(e)}"

# --- Main Application UI ---
def main():
    # Language selection in the sidebar
    languages = {
        "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de",
        "Chinese (Simplified)": "zh-CN", "Arabic": "ar", "Russian": "ru", "Japanese": "ja"
    }
    st.sidebar.title("⚙️ " + translate_text("Settings", "en"))
    selected_language_name = st.sidebar.selectbox(
        translate_text("Select Language:", "en"),
        list(languages.keys())
    )
    language_code = languages[selected_language_name]

    # Main page title and introduction
    st.title("🩺 " + translate_text("AI Health Assistant", language_code))
    st.write(translate_text(
        "Welcome! I am here to provide general health information and analyze your symptoms. "
        "Please remember, I am not a substitute for a real doctor.", language_code
    ))

    # Tabbed interface for different features
    tab1, tab2 = st.tabs([
        " S" + translate_text("ymptom Checker", language_code),
        " " + translate_text("General Health Q&A", language_code)
    ])

    # == Symptom Checker Tab ==
    with tab1:
        st.header(translate_text("Symptom Analysis Form", language_code))
        st.info(translate_text(
            "Please provide your details and symptoms below for a preliminary AI-powered analysis.", language_code
        ))

        with st.form("symptom_form"):
            st.subheader(translate_text("Basic Information", language_code))
            name = st.text_input(translate_text("Name:", language_code))
            age = st.number_input(translate_text("Age:", language_code), min_value=0, max_value=120, step=1)
            gender = st.selectbox(translate_text("Gender:", language_code),
                                  [translate_text("Male", language_code),
                                   translate_text("Female", language_code),
                                   translate_text("Prefer not to say", language_code)])

            st.subheader(translate_text("Describe Your Symptoms", language_code))
            symptoms_text = st.text_area(
                translate_text("Please describe your main symptoms, when they started, and their severity:", language_code),
                height=150
            )

            submitted = st.form_submit_button(translate_text("Analyze My Symptoms", language_code))

            if submitted:
                if not symptoms_text or not name or age <= 0:
                    st.error(translate_text("Please fill in all the fields before submitting.", language_code))
                else:
                    with st.spinner(translate_text("AI is analyzing your symptoms...", language_code)):
                        # Construct a detailed prompt for the AI
                        analysis_prompt = (
                            f"Analyze the following health case:\n"
                            f"- Patient Name: {name}\n"
                            f"- Age: {age}\n"
                            f"- Gender: {gender}\n"
                            f"- Described Symptoms: {symptoms_text}\n\n"
                            "Based on these symptoms, please provide:\n"
                            "1. A list of possible conditions (from most likely to least likely).\n"
                            "2. A brief, simple explanation for each possible condition.\n"
                            "3. A suggested course of action (e.g., rest, hydration, or see a doctor).\n"
                        )
                        analysis_response = get_ai_response(analysis_prompt, language_code)
                        st.divider()
                        st.subheader(translate_text("AI Analysis Result", language_code))
                        st.markdown(analysis_response)

    # == General Health Q&A Tab ==
    with tab2:
        st.header(translate_text("Ask a General Health Question", language_code))
        st.info(translate_text(
            "You can ask about diseases, treatments, nutrition, fitness, or any other health-related topic.", language_code
        ))
        
        user_query = st.text_input(translate_text("Your question:", language_code), key="qa_input")
        
        if user_query:
            with st.spinner(translate_text("Finding an answer...", language_code)):
                # Translate query to English for the AI model for better consistency
                english_query = translate_text(user_query, "en")
                ai_response = get_ai_response(english_query, language_code)
                st.divider()
                st.subheader(translate_text("AI Assistant's Response:", language_code))
                st.markdown(ai_response)

if __name__ == "__main__":
    main()

