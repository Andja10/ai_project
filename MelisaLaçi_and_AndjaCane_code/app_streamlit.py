import streamlit as st
import os
from projectcode import MathTutorAgent

# Load CSS from external file 
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Use it 
load_css() 
from projectcode import ( 
    load_dataset, 
    ProblemClassifier, 
    MathTutorAgent ) 
st.set_page_config(page_title="Intelligent Math Tutor", layout="centered") 

# Header

col1, col2 = st.columns([5, 1])

with col1:
    st.title("Intelligent Math Tutor")

# Load Agent 

@st.cache_resource
def load_agent():
    return MathTutorAgent()

agent = load_agent()


with st.sidebar:
    
    st.markdown("<h2>📊 Model Evaluation</h2>", unsafe_allow_html=True)
    
    # Get evaluation
    acc, f1, df = agent.evaluate()
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metricA">Accuracy: {round(acc, 3)}</div>
        <div class="metricF">F1 Score: {round(f1, 3)}</div>        
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Show Classifier Metrics", expanded=False):
        st.dataframe(df.style.format({
        "precision":"{:.2f}", 
        "recall":"{:.2f}", 
        "f1-score":"{:.2f}"
    }))
        

# Problem Input

with st.form("solve_form"):
    problem = st.text_input(
        label="Enter a math problem to see step-by-step solution:",  
        placeholder="Example: x^2 - 5*x + 6 = 0"
    )
    solve_button_clicked = st.form_submit_button("Solve")

# Solve Problem

if solve_button_clicked:
    stripped_problem = problem.strip()  

    if not stripped_problem:
        # Case 1: Empty input
        st.warning("Please enter a math problem.")
    else:
        # Case 2: Check if the expression is valid
        from projectcode import parse_expression  
        expr, detected_type = parse_expression(stripped_problem)

        expr, expr_type = parse_expression(problem)
        if expr is None or expr_type == "invalid":
            st.warning("Invalid equation or expression.")
        else:
            # Case 3: Valid input → solve
            result = agent.solve_problem(stripped_problem)

            st.subheader("Classification")
            st.write("ML Prediction:", result["ml_type"])
            st.write("Final Type:", result["final_type"])

            st.subheader("Step-by-Step Plan & Solution")
            for i, step in enumerate(result["plan"], 1):
                st.write(f"{i}. {step}")

            st.subheader("Final Answer")
            st.success(result["solution"])


