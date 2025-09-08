import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree

# --- Configuration ---
st.set_page_config(page_title="Healtho Diagnosis Bot", page_icon="🩺", layout="centered")

# --- Model and Data Loading (Cached for performance) ---
@st.cache_resource
def load_model_and_data():
    """
    Loads data, trains the decision tree model, and prepares necessary components.
    This function is cached to run only once.
    """
    # Load datasets
    training = pd.read_csv('Training.csv')
    
    # Get feature columns (all columns except the last one)
    cols = training.columns[:-1]
    
    # Prepare data
    x = training[cols]
    y = training['prognosis']
    
    # Group data for symptom lookup
    reduced_data = training.groupby(training['prognosis']).max()
    
    # Label encoding for the target variable
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)
    
    # Train the Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(x, y_encoded)
    
    return clf, le, cols, reduced_data

# Load all necessary objects
clf, le, cols, reduced_data = load_model_and_data()
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
    """Displays the current symptom question to the user."""
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

def display_report():
    """Displays the final diagnosis report."""
    # Get disease from the leaf node value
    present_disease_encoded = tree_.value[st.session_state.node][0].argmax()
    present_disease = le.inverse_transform([present_disease_encoded])[0]

    st.header("🩺 Healtho Diagnosis Report", divider='blue')
    st.success(f"**Predicted Disease:** {present_disease}")
    
    # Display confirmed symptoms
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
    except (FileNotFoundError, Exception) as e:
        st.error("Could not load doctor consultation advice.")

    if st.button("Start New Diagnosis"):
        # Reset the session state to start over
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# --- Main App Logic ---
initialize_session_state()

st.title("Healtho: Your Personal Health Assistant")

if st.session_state.page == 'home':
    st.write("Welcome! I'm here to help you with a preliminary health diagnosis based on your symptoms.")
    st.write("Please answer the following questions to the best of your ability.")
    if st.button("Start Diagnosis", type="primary"):
        st.session_state.page = 'diagnosis'
        st.rerun()

elif st.session_state.page == 'diagnosis':
    current_node = st.session_state.node
    # Check if it's a leaf node (diagnosis) or an internal node (question)
    if tree_.feature[current_node] != _tree.TREE_UNDEFINED:
        display_question(current_node)
    else:
        # Reached a leaf node, show the report
        st.session_state.page = 'report'
        st.rerun()

elif st.session_state.page == 'report':
    display_report()
