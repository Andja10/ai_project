import sympy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import random
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report


# ===============================
# Large Synthetic Dataset
# ===============================
def load_dataset():
    X = []
    y = []

    # --------------------
    # Linear equations
    # --------------------
    for a in range(1, 6):
        for b in range(0, 6):
            X.append(f"{a}*x + {b} = {b + a}")
            y.append("linear")
            X.append(f"{b} + {a}*x = {b + a}")
            y.append("linear")
    
    # --------------------
    # Quadratic equations
    # --------------------
    quadratics = [
        "x^2 - 5*x + 6 = 0",
        "x^2 + 4*x + 4 = 0",
        "2*x^2 - 8 = 0",
        "x^2 = 9",
        "3*x^2 + x - 10 = 0",
        "x^2 - 7*x = 0"
    ]
    for q in quadratics:
        X.append(q)
        y.append("quadratic")
    
    # Add synthetic variations
    for a in range(1, 4):
        for b in range(1, 4):
            X.append(f"{a}*x^2 + {b}*x - {a*b} = 0")
            y.append("quadratic")
    
    # --------------------
    # Cubic and higher polynomials
    # --------------------
    polynomials = [
        "x^3 - 1 = 0",
        "x^3 + 3*x^2 - 4 = 0",
        "x^4 - x^3 = 2",
        "x^4 - 16 = 0",
        "2*x^3 - 8 = 0"
    ]
    for p in polynomials:
        X.append(p)
        y.append("polynomial")
    
    # --------------------
    # Derivatives
    # --------------------
    derivatives = [
        "differentiate x^3",
        "differentiate x^2 + 3*x",
        "differentiate 5*x^4",
        "differentiate 3*x^3 - 7"
    ]
    for d in derivatives:
        X.append(d)
        y.append("derivative")
    
    # --------------------
    # Simplification 
    # --------------------
    simplifications = [
        "3*(x + 2) - x",
        "2*(x - 3) + 4",
        "5*x - 2*x + 7",
        "4*(x + 1) - 2*(x - 1)",
        "x + x + x",
        "6 - (2 + x)"
    ]
    for s in simplifications:
        X.append(s)
        y.append("simplification")
    
    return X, y


# ===============================
# Problem Classifier
# ===============================

class ProblemClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2))
        self.model = LinearSVC()
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        self.X_vec = X_vec

    def predict(self, text):
        X_vec = self.vectorizer.transform([text])
        return self.model.predict(X_vec)[0]


# ===============================
# Evaluation inside agent
# ===============================

def evaluate_classifier(classifier):
    y_true = classifier.y_train
    y_pred = classifier.model.predict(classifier.X_vec)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    return acc, f1, df 

# ===============================
# Utility Functions
# ===============================

def parse_expression(problem, allowed_vars={"x"}):
    """Return a sympy expression and type. Returns 'invalid' if input is not valid."""
    problem = problem.replace("^", "**").strip()
    
    # Reject empty input
    if not problem:
        return None, "invalid"
    
    try:
        # Handle derivatives
        if problem.lower().startswith("differentiate") or problem.lower().startswith("d/dx"):
            expr_text = problem.lower().replace("differentiate","").replace("d/dx","").strip()
            
            # Reject if derivative contains '='
            if "=" in expr_text:
                return None, "invalid"
            
            expr = sp.sympify(expr_text)
            # Check if it contains at least one allowed variable
            if not expr.free_symbols.intersection({sp.symbols(v) for v in allowed_vars}):
                return None, "invalid"
            
            return expr, "derivative"
        
        # Handle equations
        if "=" in problem:
            left, right = problem.split("=")
            expr = sp.sympify(left) - sp.sympify(right)
            # Reject if both sides have no allowed variables or numbers
            if not expr.free_symbols.intersection({sp.symbols(v) for v in allowed_vars}) and not expr.atoms(sp.Number):
                return None, "invalid"
            return expr, "equation"
        
        # Handle general expressions
        expr = sp.sympify(problem)
        if not expr.free_symbols.intersection({sp.symbols(v) for v in allowed_vars}) and not expr.atoms(sp.Number):
            return None, "invalid"
        
        return expr, "expression"
    
    except (sp.SympifyError, SyntaxError):
        return None, "invalid"



def detect_variable(expr):
    vars_ = list(expr.free_symbols)
    return vars_[0] if vars_ else None

def expr_to_text(expr):
    return str(expr).replace("**", "^")

def filter_real_solutions(solutions):
    real = []
    for s in solutions:
        if s.is_real or s.is_Number:
            real.append(sp.simplify(s))
    return real

def format_numeric_solution(s):
    s_val = float(s.evalf())
    if s_val.is_integer():
        return int(s_val)
    return round(s_val,4)

# ===============================
# Solver
# ===============================
class MathSolver:
    def solve_equation(self, expr, var):
        steps = []
        current_expr = expr
        steps.append(f"Original equation: {expr_to_text(current_expr)} = 0")

        simplified = sp.simplify(current_expr)
        if simplified != current_expr:
            current_expr = simplified
            steps.append(f"Simplify: {expr_to_text(current_expr)} = 0")

        numeric_expr = current_expr
        for _ in range(10):
            new_expr = numeric_expr.xreplace({n:n.evalf() for n in numeric_expr.atoms(sp.Number)})
            if new_expr == numeric_expr:
                break
            numeric_expr = new_expr
            steps.append(f"Simplify numeric operations: {expr_to_text(numeric_expr)} = 0")

        factored = sp.factor(numeric_expr)
        if factored != numeric_expr:
            numeric_expr = factored
            steps.append(f"Factor: {expr_to_text(numeric_expr)} = 0")

        solutions = sp.solve(numeric_expr,var)
        real_solutions = filter_real_solutions(solutions)
        if real_solutions:
            numeric_solutions = [format_numeric_solution(s) for s in real_solutions]
            steps.append(f"Solve for {var}: {numeric_solutions}")
            return steps, numeric_solutions
        else:
            steps.append("Solve for variable: no real solutions")
            return steps,"No real solution"

    def solve_derivative(self, expr, var):
        steps=[]
        steps.append(f"Original function: {expr_to_text(expr)}")
        deriv = sp.diff(expr,var)
        steps.append(f"Differentiated: {expr_to_text(deriv)}")
        return steps,deriv
    
    def solve_simplification(self, expr):
        steps = []
        steps.append(f"Original expression: {expr_to_text(expr)}")
        simplified = sp.simplify(expr)
        if simplified != expr:
            steps.append(f"Simplified: {expr_to_text(simplified)}")
        return steps, simplified



# ===============================
# Planner
# ===============================
class SolutionPlanner:
    def generate_plan(self, problem_type):
        if problem_type=="derivative":
            return [
                "Identify variable of differentiation",
                "Apply differentiation rules",
                "Simplify the result"
            ]
        return [
            "Move all terms to one side",
            "Simplify equation",
            "Solve for the unknown variable"
        ]

# ===============================
# Math Tutor Agent
# ===============================
class MathTutorAgent:
    def __init__(self):
        X,y = load_dataset()
        self.classifier = ProblemClassifier()
        self.classifier.train(X,y)
        self.solver = MathSolver()
        self.planner = SolutionPlanner()

    def solve_problem(self, problem):
        # First, try to parse the expression
        expr, detected_type = parse_expression(problem)
        
        if detected_type == "invalid" or expr is None:
            # Immediately return if invalid — no classification or solving
            return {
                "ml_type": "unknown",
                "final_type": "invalid",
                "plan": [],
                "solution": "Invalid equation or expression."
            }
        
        # Now classify (ML) only if valid
        ml_type = self.classifier.predict(problem)
        
        if detected_type == "derivative":
            steps, solution = self.solver.solve_derivative(expr, sp.symbols('x'))
            final_type = "derivative"
        elif detected_type == "expression":
            steps, solution = self.solver.solve_simplification(expr)
            final_type = "simplification"
        else:
            var = detect_variable(expr)
            if var is None:
                return {
                    "ml_type": ml_type,
                    "final_type": "unknown",
                    "plan": [],
                    "solution": "No variable detected."
                }

            degree = sp.degree(expr, var)
            final_type = ml_type
            if degree == 1: final_type = "linear"
            elif degree == 2: final_type = "quadratic"
            elif degree and degree > 2: final_type = "polynomial"

            steps, solution = self.solver.solve_equation(expr, var)

        return {
            "ml_type": ml_type,
            "final_type": final_type,
            "plan": steps,
            "solution": solution
        }



    def evaluate(self):
        return evaluate_classifier(self.classifier)
