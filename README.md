# Neural Network-based Multi-Objective Evolutionary Algorithm for Dynamic Workflow Scheduling in Cloud Computing

This project implements a sophisticated workflow scheduling system for cloud computing environments, combining evolutionary algorithms with neural networks to optimize multiple competing objectives: makespan (total execution time), cost, and reliability.

## Overview

The system uses an NSGA-II (Non-dominated Sorting Genetic Algorithm) enhanced with neural network fitness prediction to find Pareto-optimal solutions to the multi-objective workflow scheduling problem. By leveraging machine learning, the algorithm can more efficiently explore the solution space and converge faster toward optimal solutions.

Key features:
- Multi-objective optimization via NSGA-II algorithm
- Neural network fitness prediction to accelerate evaluations
- Support for arbitrary directed acyclic graph (DAG) workflows
- Dynamic resource allocation in heterogeneous cloud environments
- Advanced visualization of Pareto fronts and schedules

## System Architecture

The system consists of several major components:

1. **Workflow Model**: Manages tasks, their dependencies, and constraints
2. **Cloud Environment**: Represents VMs with varied resources and pricing
3. **NSGA-II Algorithm**: Handles the evolutionary optimization process
4. **Neural Network Model**: Accelerates fitness evaluation through prediction
5. **Visualization**: Provides insights into scheduling decisions

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Setup

The provided `setup.sh` script will install all necessary dependencies and set up the project structure. Run:

```bash
chmod +x setup.sh
./setup.sh
```

This will:
1. Install Python and required packages
2. Create a virtual environment
3. Set up the project directory structure
4. Insert the main code into the appropriate files

## Usage

### Basic Usage

1. After running the setup script, navigate to the project directory:
   ```bash
   cd workflow_scheduler
   ```

2. Activate the virtual environment:
   ```bash
   source ../workflow_scheduler_env/bin/activate
   ```

3. Run the example:
   ```bash
   ./run_example.py
   ```

### Advanced Usage

Use the `launch.sh` script to run the scheduler with custom parameters:

```bash
# Run with 30 tasks, 8 VMs, population size 100, and 150 generations
./launch.sh --tasks 30 --vms 8 --population 100 --generations 150
```

## Code Structure

### Core Classes

- **Task**: Represents a computational task with dependencies and resource requirements
- **VirtualMachine**: Models cloud resources with specific CPU, memory, and cost characteristics
- **CloudEnvironment**: Manages VMs and network connections
- **Workflow**: Maintains a collection of tasks and their dependencies
- **Individual**: Represents a solution (mapping of tasks to VMs)
- **NeuralNetworkModel**: Predicts fitness values for solutions
- **NNSGA2**: The multi-objective optimization algorithm

### Important Functions

- `create_synthetic_workflow()`: Generates test workflows with random properties
- `create_cloud_environment()`: Sets up a test cloud environment
- `save_results()`: Exports optimization results to CSV
- `visualize_schedule()`: Creates a Gantt chart of the scheduling solution
- `plot_pareto_front()`: Visualizes the trade-offs between objectives

## Multi-Objective Optimization

The system optimizes for three competing objectives:

1. **Makespan**: The total time to complete all tasks (minimize)
2. **Cost**: The total cost of resource usage (minimize)
3. **Reliability**: A measure of load distribution quality (maximize)

Instead of producing a single solution, the algorithm finds a Pareto front of non-dominated solutions, each representing a different trade-off between the objectives.

## Neural Network Enhancement

The neural network accelerates the evolutionary process by:

1. Learning the relationship between task-VM assignments and resulting performance
2. Predicting fitness values without expensive simulation
3. Falling back to actual calculation when high accuracy is needed

This hybrid approach maintains solution quality while significantly reducing computation time.

## Results Interpretation

After running the algorithm, you'll find:

- `pareto_front_results.csv`: A detailed list of solutions with their objective values
- `pareto_front.png`: Visualization showing the trade-offs between objectives
- `best_schedule.png`: A Gantt chart showing task scheduling for the balanced solution

## Further Development

Potential areas for enhancement:
- Support for real-time workflow changes
- Integration with actual cloud APIs (AWS, Azure, GCP)
- Improved neural network architectures (transformer models, etc.)
- Energy consumption as an additional objective
- Fault tolerance and reliability modeling

## References

1. Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.
2. Zhan, Z. H., et al. (2020). Cloud workflow scheduling: A survey.
3. Durillo, J. J., & Prodan, R. (2014). Multi-objective workflow scheduling in Amazon EC2.
4. Rodriguez, M. A., & Buyya, R. (2018). Scientific workflow management systems for cloud computing.
5. Haider, M., et al. (2022). Machine learning for cloud resource management.
