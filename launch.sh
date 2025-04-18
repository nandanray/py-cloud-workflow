#!/bin/bash

# Launch script for Neural Network-based Cloud Workflow Scheduler
#
# This script activates the virtual environment and runs the workflow scheduler
# with customizable parameters

# Default parameter values
NUM_TASKS=20
NUM_VMS=5
POPULATION_SIZE=50
GENERATIONS=100

# Help message
function show_help {
    echo "Neural Network Cloud Workflow Scheduler"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -t, --tasks N         Number of tasks in the workflow (default: 20)"
    echo "  -v, --vms N           Number of VMs in the cloud environment (default: 5)"
    echo "  -p, --population N    Population size for the evolutionary algorithm (default: 50)"
    echo "  -g, --generations N   Number of generations to evolve (default: 100)"
    echo "  -h, --help            Show this help message"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -t|--tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        -v|--vms)
            NUM_VMS="$2"
            shift 2
            ;;
        -p|--population)
            POPULATION_SIZE="$2"
            shift 2
            ;;
        -g|--generations)
            GENERATIONS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if virtual environment exists
if [ ! -d "workflow_scheduler_env" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run the setup script first: ./setup.sh"
    exit 1
fi

# Check if project directory exists
if [ ! -d "workflow_scheduler" ]; then
    echo "Error: Project directory not found!"
    echo "Please run the setup script first: ./setup.sh"
    exit 1
fi

# Activate the virtual environment
source workflow_scheduler_env/bin/activate

# Create custom runner script
cat > workflow_scheduler/run_custom.py << EOF
#!/usr/bin/env python3

import sys
import random
import numpy as np
import tensorflow as tf
from cloud_workflow_scheduler import (
    create_synthetic_workflow, 
    create_cloud_environment, 
    NNSGA2, 
    save_results, 
    visualize_schedule
)

# Set parameters
NUM_TASKS = $NUM_TASKS
NUM_VMS = $NUM_VMS
POPULATION_SIZE = $POPULATION_SIZE
GENERATIONS = $GENERATIONS

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print("Creating workflow with \${NUM_TASKS} tasks...")
workflow = create_synthetic_workflow(num_tasks=NUM_TASKS, max_dependencies=min(5, NUM_TASKS // 4))

print("Creating cloud environment with \${NUM_VMS} VMs...")
cloud_env = create_cloud_environment(num_vms=NUM_VMS)

print("Initializing NNSGA2 optimizer with population size \${POPULATION_SIZE}...")
nnsga2 = NNSGA2(workflow, cloud_env, population_size=POPULATION_SIZE, nn_input_size=100)

print("Starting evolution for \${GENERATIONS} generations...")
pareto_front = nnsga2.evolve(generations=GENERATIONS)

# Get balanced solution
balanced = nnsga2.get_best_solution(priority="balanced")

# Print results
print("\nBest balanced solution:")
print(f"Makespan: {balanced.fitness['makespan']:.2f}")
print(f"Cost: {balanced.fitness['cost']:.2f}")
print(f"Reliability: {balanced.fitness['reliability']:.4f}")

# Save results
results_file = f"results/custom_run_t{NUM_TASKS}_v{NUM_VMS}_p{POPULATION_SIZE}_g{GENERATIONS}.csv"
save_results(pareto_front, results_file)

# Plot Pareto front
nnsga2.plot_pareto_front()

# Visualize schedule
schedule_file = f"results/custom_schedule_t{NUM_TASKS}_v{NUM_VMS}.png"
visualize_schedule(balanced, schedule_file)

print("\nExecution complete!")
EOF

# Make it executable
chmod +x workflow_scheduler/run_custom.py

# Run the custom script
echo "Running workflow scheduler with parameters:"
echo "- Tasks: $NUM_TASKS"
echo "- VMs: $NUM_VMS"
echo "- Population size: $POPULATION_SIZE"
echo "- Generations: $GENERATIONS"
echo ""

cd workflow_scheduler
python3 run_custom.py