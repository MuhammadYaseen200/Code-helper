import os
import torch
import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load CodeBERT model for code analysis
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base")

# Function to analyze code
def analyze_code(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    if predicted_class == 0:
        return "No syntax errors found. All good!"
    elif predicted_class == 1:
        return "There seems to be a syntax error. Check your code!"
    else:
        return "Code analysis inconclusive."

# Function to check for missing libraries based on import statements
def check_missing_libraries(code):
    missing_libraries = []
    import_statements = [line for line in code.splitlines() if line.startswith("import") or line.startswith("from")]
    
    # Simulated library checking
    for statement in import_statements:
        library = statement.split()[1] if "import" in statement else statement.split()[1].split('.')[0]
        if library not in ["os", "sys"]:  # Add libraries you know are installed
            missing_libraries.append(library)
    
    return missing_libraries

# Function to provide deployment suggestions
def deployment_suggestions(directory):
    files = os.listdir(directory)
    suggestions = []
    if "requirements.txt" not in files:
        suggestions.append("Create a requirements.txt file for dependencies.")
    if not any(file.endswith(".py") for file in files):
        suggestions.append("No Python files found in the directory.")
    if "app.py" not in files:
        suggestions.append("Consider renaming your main file to app.py for Streamlit.")
    
    return suggestions

# Streamlit app
def main():
    st.title("Code Analysis App")

    # User input for code analysis
    code_input = st.text_area("Paste your code here:", height=200)
    
    if st.button("Analyze Code"):
        analysis_result = analyze_code(code_input)
        st.subheader("Code Analysis Result:")
        st.write(analysis_result)

        # Check for missing libraries
        missing_libs = check_missing_libraries(code_input)
        if missing_libs:
            st.subheader("Missing Libraries:")
            st.write(", ".join(missing_libs))
        else:
            st.write("No missing libraries found.")

    # User input for project directory
    project_directory = st.text_input("Enter the path to your project directory:")
    
    if st.button("Get Deployment Suggestions"):
        if os.path.exists(project_directory):
            suggestions = deployment_suggestions(project_directory)
            if suggestions:
                st.subheader("Deployment Suggestions:")
                for suggestion in suggestions:
                    st.write(f"- {suggestion}")
            else:
                st.write("Your project directory is ready for deployment!")
        else:
            st.write("Invalid directory path. Please check and try again.")

if __name__ == "__main__":
    main()
