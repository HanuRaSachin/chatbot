import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings for a cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")


# --- App Configuration ---
st.set_page_config(page_title="Healtho Diagnosis Bot", page_icon="🩺", layout="centered")

# --- Model and Data Loading (Cached for performance) ---
@st.cache_resource
def load_model_and_data():
    """
    Loads data, trains the decision tree model, and prepares necessary components.
    This function is cached, so it only runs once when the app starts.
    """
    try:
        # Load datasets
        training = pd.read_csv('Training.csv')
        
        # Get feature columns (all columns except the last one, 'prognosis')
        cols = training.columns[:-1]
        
        # Prepare data
        x = training[cols]
        y = training['prognosis']
        
        # Group data for symptom lookup later
        reduced_data = training.groupby(training['prognosis']).max()
        
        # Label encoding for the target variable 'prognosis'
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y_encoded = le.transform(y)
        
        # Train the Decision Tree Classifier
        clf = DecisionTreeClassifier()
        clf.fit(x, y_encoded)
        
        # Return the original training dataframe as well for visualization
        return clf, le, cols, reduced_data, training
    except FileNotFoundError as e:
        st.error(f"Error loading data files: {e}. Make sure 'Training.csv' is in the same directory as the app.")
        return None, None, None, None, None

# Load all necessary objects
clf, le, cols, reduced_data, training_df = load_model_and_data()

# Check if model loading was successful before proceeding
if clf is not None:
    tree_ = clf.tree_
    feature_names = [cols[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    if 'node' not in st.session_state:
        st.session_state.node = 0
    if 'symptoms_present' not in st.session_state:
        st.session_state.symptoms_present = []
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

# --- UI Functions ---
def display_question(node):
    """Displays the current symptom question and buttons."""
    symptom_name = feature_names[node].replace('_', ' ')
    st.subheader(f"Do you have {symptom_name}?", divider='rainbow')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Yes", use_container_width=True, type="primary"):
            st.session_state.symptoms_present.append(feature_names[node])
            st.session_state.node = tree_.children_right[node]
            st.rerun()
            
    with col2:
        if st.button("No", use_container_width=True):
            st.session_state.node = tree_.children_left[node]
            st.rerun()

def display_symptom_chart(disease, training_df):
    """Creates and displays a bar chart for symptom prevalence."""
    st.subheader("Symptom Prevalence for this Condition", divider='gray')
    st.info("This chart shows the most common symptoms associated with the predicted condition, based on the training data.")
    
    # Filter symptoms for the predicted disease from the original training data
    symptoms = training_df.loc[training_df['prognosis'] == disease, training_df.columns[:-1]]
    
    if not symptoms.empty:
        # Calculate the count of each symptom
        symptom_counts = symptoms.sum().sort_values(ascending=False)
        top_symptoms = symptom_counts[symptom_counts > 0].head(10) # Get top 10 non-zero symptoms

        if not top_symptoms.empty:
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=top_symptoms.values, y=top_symptoms.index, palette='viridis', ax=ax)
            ax.set_title(f'Most Common Symptoms for {disease}', fontsize=16)
            ax.set_xlabel('Prevalence Count in Training Data', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            
            # Use Streamlit to display the plot
            st.pyplot(fig)
        else:
            st.write("No specific symptom data to visualize for this condition.")
    else:
        st.write("Could not retrieve symptom data for visualization.")


def display_report(training_df):
    """Displays the final diagnosis report."""
    # Get disease from the leaf node value
    present_disease_encoded = tree_.value[st.session_state.node][0].argmax()
    present_disease = le.inverse_transform([present_disease_encoded])[0]

    st.header("🩺 Healtho Diagnosis Report", divider='blue')
    st.success(f"**Predicted Disease:** {present_disease}")
    
    # Display symptoms the user confirmed
    confirmed_symptoms_str = ", ".join([s.replace('_', ' ') for s in st.session_state.symptoms_present])
    st.write(f"**Symptoms You Confirmed:** {confirmed_symptoms_str if confirmed_symptoms_str else 'None'}")
    
    # Display other possible symptoms for the diagnosed disease
    st.subheader("Other Possible Symptoms:")
    symptoms_given = reduced_data.columns[reduced_data.loc[present_disease].values[0].nonzero()]
    for i, symptom in enumerate(symptoms_given):
        st.markdown(f"- {symptom.replace('_', ' ')}")
    
    # Doctor consultation recommendation
    try:
        import csv
        with open('doc_consult.csv', 'r') as f:
            read = csv.reader(f)
            consult = {rows[0]: int(rows[1]) for rows in read}
        
        risk_level = consult.get(present_disease, 0)
        st.subheader("Doctor Consultation Advice:", divider='gray')
        if risk_level > 50:
            st.warning("**High Risk:** You should consult a doctor as soon as possible.")
        else:
            st.info("**Moderate Risk:** You may want to consult a doctor for a professional opinion.")
    except (FileNotFoundError, Exception):
        st.error("Could not load doctor consultation advice. Make sure 'doc_consult.csv' is present.")

    # Display the symptom prevalence chart
    display_symptom_chart(present_disease, training_df)

    if st.button("Start New Diagnosis"):
        # Reset the session state to start over
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Main App Logic ---
# Initialize session state for the first run
initialize_session_state()

st.title("Healtho: Your Personal Health Assistant")

# Main router: directs to the correct page based on session state
if clf is not None:
    if st.session_state.page == 'home':
        st.write("Welcome! I'm here to help with a preliminary health diagnosis based on your symptoms.")
        st.write("Please answer the questions to the best of your ability. This is not a substitute for professional medical advice.")
        if st.button("Start Diagnosis", type="primary"):
            st.session_state.page = 'diagnosis'
            st.rerun()

    elif st.session_state.page == 'diagnosis':
        current_node = st.session_state.node
        # Check if it's a leaf node (diagnosis) or an internal node (question)
        if tree_.feature[current_node] != _tree.TREE_UNDEFINED:
            display_question(current_node)
        else:
            st.session_state.page = 'report'
            st.rerun()

    elif st.session_state.page == 'report':
        display_report(training_df)
else:
    st.error("Application cannot start because the model failed to load. Please check the data files.")

