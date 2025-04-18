#!/bin/bash

# Setup script for Neural Network-based Multi-Objective Evolutionary Algorithm
# for Dynamic Workflow Scheduling in Cloud Computing
#
# For Ubuntu 22.04 LTS

set -e  # Exit immediately if a command exits with a non-zero status

echo "---------------------------------------------"
echo "Setting up Neural Network Cloud Workflow Scheduler"
echo "---------------------------------------------"

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install Python and pip if not already installed
echo "Installing Python and pip..."
sudo apt-get install -y python3 python3-pip python3-venv

# Create a virtual environment
echo "Creating Python virtual environment..."
python3 -m venv workflow_scheduler_env

# Activate the virtual environment
echo "Activating virtual environment..."
source workflow_scheduler_env/bin/activate

# Install required Python packages
echo "Installing required Python packages..."
pip install --upgrade pip
pip install numpy pandas matplotlib tensorflow scikit-learn
#pip install -r requirements.txt #uncomment this if the above line fails to install

# Create project directory structure
echo "Creating project directory structure..."
mkdir -p workflow_scheduler/data
mkdir -p workflow_scheduler/results
mkdir -p workflow_scheduler/models

# Create example usage script
cat > workflow_scheduler/run_example.py << 'EOF'
#!/usr/bin/env python3

# Example usage of Cloud Workflow Scheduler
from cloud_workflow_scheduler import main

if __name__ == "__main__":
    main()
EOF

# Make it executable
chmod +x workflow_scheduler/run_example.py

# Save the main script
echo "Saving main script..."
cat > workflow_scheduler/cloud_workflow_scheduler.py << 'EOF'
# Import the content of the main script here
# (This will be filled by the sed command below)
EOF

# Use sed to insert the main script content into the created file
# Replace this line with the actual code insertion when you have the full code

echo "----------------------------------------"
echo "Setup complete! Follow these steps to run the code:"
echo "1. cd workflow_scheduler"
echo "2. source ../workflow_scheduler_env/bin/activate"
echo "3. ./run_example.py"
echo "----------------------------------------"

# Create a helper script to insert the main code
cat > insert_code.sh << 'EOF'
#!/bin/bash
# Insert the code from the first script into the main Python file
sed -i '1d' workflow_scheduler/cloud_workflow_scheduler.py  # Remove the first line (comment)
cat cloud_workflow_scheduler_code.py > workflow_scheduler/cloud_workflow_scheduler.py
echo "Code inserted successfully!"
EOF

chmod +x insert_code.sh

# Create a placeholder for the main code
touch cloud_workflow_scheduler_code.py

echo "Please paste the main code into the file 'cloud_workflow_scheduler_code.py' and then run './insert_code.sh'"