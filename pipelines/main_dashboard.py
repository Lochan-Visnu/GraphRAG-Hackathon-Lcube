import tkinter as tk
from tkinter import messagebox
import subprocess
import os

class L3MedicalDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("L^3 Educations: Medical GraphRAG Benchmark")
        self.root.geometry("600x450")
        self.root.configure(bg="#f0f4f7")

        # Title
        tk.Label(root, text="Medical Inference Benchmark", font=("Arial", 18, "bold"), bg="#f0f4f7", fg="#2c3e50").pack(pady=20)
        tk.Label(root, text="Compare Token Efficiency & Accuracy", font=("Arial", 10), bg="#f0f4f7", fg="#7f8c8d").pack()

        # Buttons Frame
        btn_frame = tk.Frame(root, bg="#f0f4f7")
        btn_frame.pack(pady=30)

        # Pipeline 1 Button
        self.btn_p1 = tk.Button(btn_frame, text="Run Pipeline 1 (LLM-Only)", width=30, height=2, 
                               command=lambda: self.run_script("P1_UI.py"), bg="#3498db", fg="white", font=("Arial", 10, "bold"))
        self.btn_p1.grid(row=0, column=0, pady=10)

        # Pipeline 2 Button
        self.btn_p2 = tk.Button(btn_frame, text="Run Pipeline 2 (Basic RAG)", width=30, height=2, 
                               command=lambda: self.run_script("P2_UI.py"), bg="#9b59b6", fg="white", font=("Arial", 10, "bold"))
        self.btn_p2.grid(row=1, column=0, pady=10)

        # Pipeline 3 Button
        self.btn_p3 = tk.Button(btn_frame, text="Run Pipeline 3 (GraphRAG)", width=30, height=2, 
                               command=lambda: self.run_script("P3_UI.py"), bg="#e67e22", fg="white", font=("Arial", 10, "bold"))
        self.btn_p3.grid(row=2, column=0, pady=10)

        # Footer
        tk.Label(root, text="Built for TigerGraph GraphRAG Hackathon", font=("Arial", 8, "italic"), bg="#f0f4f7").pack(side="bottom", pady=10)

    def run_script(self, script_name):
        if os.path.exists(script_name):
            try:
                subprocess.Popen(["python", script_name])
            except Exception as e:
                messagebox.showerror("Error", f"Could not launch {script_name}: {e}")
        else:
            messagebox.showwarning("Not Found", f"{script_name} not found in the directory.\\n\\nPlease ensure the file is uploaded.")

if __name__ == "__main__":
    root = tk.Tk()
    app = L3MedicalDashboard(root)
    root.mainloop()
