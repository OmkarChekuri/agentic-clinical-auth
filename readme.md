# **Local Setup Guide: Agentic Clinical Auth**

Follow these steps to deploy the PenguinAI Healthcare engine and the React dashboard on your local environment.

### **1\. Prerequisites**

* **Python 3.10 or higher**: Ensure Python is installed (python \--version).  
* **Ollama (Optional)**: If you plan to run models locally (e.g., llama3.2), download and install it from [ollama.com](https://ollama.com).  
* **Gemini API Key (Optional)**: If using cloud reasoning, obtain a key from the [Google AI Studio](https://aistudio.google.com/).

### **2\. Project Initialization**

Create your project directory and the necessary subfolders:

\# Create directory structure  
mkdir agentic-clinical-auth  
cd agentic-clinical-auth  
mkdir policies

### **3\. Environment Setup**

Create a virtual environment to isolate your dependencies:

\# Create virtual environment  
python \-m venv venv

\# Activate it (Windows)  
.\\venv\\Scripts\\activate

\# Activate it (Mac/Linux)  
\# source venv/bin/activate

### **4\. Install Dependencies**

Install the required Python packages using the requirements.txt file we created:

pip install \-r requirements.txt

### **5\. Configuration**

Ensure your .env file is in the root directory. If you are using Ollama, make sure the service is running and you have pulled the model:

\# Pull the required local model  
ollama pull llama3.2

### **6\. Run the Backend**

Start the FastAPI server. This initializes the agentic system and the lifespan resources:

\# Run the application  
python main.py

*The server will be live at http://localhost:8000.*

### **7\. Launch the Frontend**

The dashboard is a self-contained React application. You can run it in two ways:

* **Simple**: Right-click index.html and open it in your browser.  
* **Developer Mode**: Use a local server (like the "Live Server" extension in VS Code) to serve index.html to avoid potential CORS edge cases.

### **8\. Verification**

1. Open the dashboard in your browser.  
2. Ensure the "Live Node" indicator in the top right is green (meaning it successfully reached the backend).  
3. Submit a test case with Patient ID P123 to see the agents begin their reasoning loop.