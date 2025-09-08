import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Importing the dataset
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')
# Saving the information of columns
cols = training.columns
cols = cols[:-1]
# Slicing and Dicing the dataset to separate features from predictions
x = training[cols]
y = training['prognosis']

# Dimensionality Reduction for removing redundancies
reduced_data = training.groupby(training['prognosis']).max()

# Encoding/mapping String values to integer constants
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Splitting-the-dataset-into-training-set-and-test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Implement the Decision-Tree-Classifier
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

# Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad you are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return disease[0] # Return the string directly

def tree_to_code(tree, feature_names):
    """
    Outputs a decision tree traversal for diagnosis.
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    symptoms_present = []

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Continuously ask until a valid 'yes' or 'no' is received
            while True:
                print(f"\nHealtho: Do you have {name.replace('_', ' ')}? (yes/no)")
                ans = input("You: ").lower()
                if ans in ['yes', 'no']:
                    break
                else:
                    print("Healtho: I'm sorry, I don't understand. Please answer with 'yes' or 'no'.")

            if ans == 'yes':
                val = 1
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
            else: # ans == 'no'
                val = 0
                recurse(tree_.children_left[node], depth + 1)
        else:
            # Reached a leaf node (final diagnosis)
            present_disease = print_disease(tree_.value[node])
            
            # Get other symptoms for this disease
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            
            # --- Display Final Report ---
            print("\n" + "="*30)
            print("HEALTHO DIAGNOSIS REPORT")
            print("="*30)
            
            print(f"Predicated Disease: {present_disease}")
            
            # Display symptoms the user confirmed
            confirmed_symptoms = ", ".join([s.replace('_', ' ') for s in symptoms_present])
            print(f"Symptoms You Confirmed: {confirmed_symptoms if confirmed_symptoms else 'None'}")
            
            # Display other possible symptoms
            print("\nOther Possible Symptoms:")
            for i, symptom in enumerate(symptoms_given):
                print(f"{i+1}. {symptom.replace('_', ' ')}")
            
            # Doctor consultation recommendation
            try:
                import csv
                with open('doc_consult.csv', 'r') as f:
                    read = csv.reader(f)
                    consult = {rows[0]: int(rows[1]) for rows in read}
                
                risk_level = consult.get(present_disease, 0) # Default to 0 if not found
                print("\n" + "-"*30)
                if risk_level > 50:
                    print("Risk Level: High Risk")
                    print("Recommendation: You should consult a doctor as soon as possible.")
                else:
                    print("Risk Level: Moderate Risk")
                    print("Recommendation: You may want to consult a doctor for a professional opinion.")
                print("-"*30)

            except FileNotFoundError:
                print("\n[Warning] 'doc_consult.csv' not found. Cannot provide doctor consultation advice.")
            except Exception as e:
                print(f"\n[Error] Could not process doctor consultation data: {e}")


    recurse(0, 1)

# --- Main Interaction Loop ---
if __name__ == "__main__":
    flag = True
    print("Healtho: My name is Healtho. I'm a chatbot designed to help with potential health issues.")
    print("Healtho: To start a diagnosis, say 'hi' or 'hello'. To exit, type 'bye'.")

    while flag:
        print("\nYour turn:")
        user_response = input().lower()

        if user_response in ['bye', 'exit', 'quit']:
            flag = False
            print("Healtho: Bye! Take care.")
        elif user_response in ['thanks', 'thank you']:
            flag = False
            print("Healtho: You are welcome! Stay healthy.")
        elif greeting(user_response) is not None:
            print(f"Healtho: {greeting(user_response)}! Let's start the diagnosis.")
            print("Healtho: Please answer the following questions about your symptoms.")
            tree_to_code(clf, cols)
        else:
            print("Healtho: I'm sorry, I don't understand. Please say 'hello' to start a diagnosis or 'bye' to exit.")
