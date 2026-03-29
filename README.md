# Resume Bias Detection & Injection (IAPC)

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Setup

1. **Navigate to the project directory:**
  

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```
   
   Activate the virtual environment:
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Data Cleaning
```bash
python src/data_cleaning.py 
```

### Bias Injection
```bash
python src/bias_injection.py
```
