**📦 Using Anaconda**

# 1. Create a new Conda environment with Python 3.10
conda create --name myenv python=3.10

# 2. Activate the environment
conda activate myenv

# 3. Install required packages
pip install -r requirements.txt

# 4. Run the application
python app.py


**🐍 Using Python (venv)**

# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment (Windows)
venv\Scripts\activate

#    On macOS/Linux, use:
# source venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Run the application
python app.py
