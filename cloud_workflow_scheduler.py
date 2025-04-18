import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle
import os
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class Task:
    def __init__(self, task_id, instructions, input_size, output_size, complexity):
        self.task_id = task_id
        self.instructions = instructions  # CPU instructions required
        self.input_size = input_size  # MB
        self.output_size = output_size  # MB
        self.complexity = complexity  # 1-10 scale
        self.dependencies = []  # List of task IDs this task depends on
        
    def add_dependency(self, task_id):
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)
            
    def __str__(self):
        return f"Task {self.task_id}: instructions={self.instructions}, input={self.input_size}MB, output={self.output_size}MB"

class VirtualMachine:
    def __init__(self, vm_id, cpu_cores, memory, cost_per_hour):
        self.vm_id = vm_id
        self.cpu_cores = cpu_cores  # Number of CPU cores
        self.memory = memory  # RAM in GB
        self.cost_per_hour = cost_per_hour  # Cost in $ per hour
        self.mips = cpu_cores * 1000  # Million Instructions Per Second
        self.current_load = 0  # Current CPU utilization (0-1)
        self.assigned_tasks = []
        
    def can_handle_task(self, task):
        # Check if VM has enough resources to handle the task
        return (self.current_load + (task.complexity / 10)) <= 1.0
        
    def assign_task(self, task):
        self.assigned_tasks.append(task)
        self.current_load += (task.complexity / 10)
        
    def calculate_task_execution_time(self, task):
        # Calculate execution time in seconds based on instructions and VM performance
        return task.instructions / (self.mips * (1 - self.current_load + 0.1))
    
    def __str__(self):
        return f"VM {self.vm_id}: {self.cpu_cores} cores, {self.memory}GB RAM, ${self.cost_per_hour}/hour, load: {self.current_load*100:.1f}%"

class CloudEnvironment:
    def __init__(self):
        self.vms = []
        self.network_bandwidth = 1000  # MB/s between VMs
        
    def add_vm(self, vm):
        self.vms.append(vm)
        
    def calculate_data_transfer_time(self, source_task, target_task):
        # Calculate time to transfer data between tasks in seconds
        return source_task.output_size / self.network_bandwidth
        
    def reset(self):
        for vm in self.vms:
            vm.current_load = 0
            vm.assigned_tasks = []

class Workflow:
    def __init__(self, workflow_id):
        self.workflow_id = workflow_id
        self.tasks = {}
        self.deadline = 0  # Deadline in seconds
        
    def add_task(self, task):
        self.tasks[task.task_id] = task
        
    def get_entry_tasks(self):
        # Return tasks that have no dependencies
        return [task for task in self.tasks.values() if not task.dependencies]
        
    def get_exit_tasks(self):
        # Return tasks that no other tasks depend on
        dependent_tasks = set()
        for task in self.tasks.values():
            for dep in task.dependencies:
                dependent_tasks.add(dep)
        
        all_task_ids = set(self.tasks.keys())
        return [self.tasks[tid] for tid in all_task_ids - dependent_tasks]
        
    def set_deadline(self, deadline):
        self.deadline = deadline

class Individual:
    def __init__(self, workflow, cloud_env):
        self.workflow = workflow
        self.cloud_env = cloud_env
        self.chromosome = {}  # task_id -> vm_id mapping
        self.fitness = {"makespan": float('inf'), "cost": float('inf'), "reliability": 0}
        self.schedule = {}  # task_id -> start_time mapping
        self.completion_times = {}  # task_id -> completion_time mapping
        
    def initialize(self):
        # Randomly assign tasks to VMs
        for task_id in self.workflow.tasks:
            self.chromosome[task_id] = random.choice(self.cloud_env.vms).vm_id
            
    def calculate_fitness(self):
        # Reset the cloud environment
        self.cloud_env.reset()
        
        # Build VM mapping
        vm_map = {vm.vm_id: vm for vm in self.cloud_env.vms}
        
        # Reset schedule and completion times
        self.schedule = {}
        self.completion_times = {}
        
        # Calculate the schedule using topological sort
        visited = set()
        scheduled = set()
        
        def schedule_task(task_id):
            if task_id in scheduled:
                return
                
            task = self.workflow.tasks[task_id]
            vm_id = self.chromosome[task_id]
            vm = vm_map[vm_id]
            
            # Ensure all dependencies are scheduled first
            for dep_id in task.dependencies:
                if dep_id not in scheduled:
                    schedule_task(dep_id)
            
            # Find the earliest time this task can start
            earliest_start_time = 0
            for dep_id in task.dependencies:
                dep_complete_time = self.completion_times[dep_id]
                data_transfer_time = self.cloud_env.calculate_data_transfer_time(
                    self.workflow.tasks[dep_id], task)
                
                # If the dependent task is on a different VM, account for data transfer
                if self.chromosome[dep_id] != vm_id:
                    dep_complete_time += data_transfer_time
                
                earliest_start_time = max(earliest_start_time, dep_complete_time)
            
            # Set start time
            self.schedule[task_id] = earliest_start_time
            
            # Calculate execution time
            exec_time = vm.calculate_task_execution_time(task)
            
            # Calculate completion time
            self.completion_times[task_id] = earliest_start_time + exec_time
            
            # Mark as scheduled
            scheduled.add(task_id)
        
        # Schedule all tasks
        for task_id in self.workflow.tasks:
            schedule_task(task_id)
        
        # Calculate makespan (completion time of the last task)
        makespan = max(self.completion_times.values())
        
        # Calculate cost - we need to know how long each VM was used
        vm_usage = {vm.vm_id: 0 for vm in self.cloud_env.vms}
        for task_id, vm_id in self.chromosome.items():
            task = self.workflow.tasks[task_id]
            vm = vm_map[vm_id]
            exec_time = vm.calculate_task_execution_time(task)
            vm_usage[vm_id] += exec_time
        
        total_cost = sum(vm_map[vm_id].cost_per_hour * (usage / 3600) for vm_id, usage in vm_usage.items())
        
        # Calculate reliability (simplified version)
        # Higher VM utilization is considered less reliable
        vm_reliability = {}
        for vm in self.cloud_env.vms:
            tasks_assigned = [task_id for task_id, assigned_vm in self.chromosome.items() if assigned_vm == vm.vm_id]
            if not tasks_assigned:
                vm_reliability[vm.vm_id] = 1.0
            else:
                total_complexity = sum(self.workflow.tasks[t].complexity for t in tasks_assigned)
                vm_reliability[vm.vm_id] = max(0, 1 - (total_complexity / (10 * len(tasks_assigned))))
        
        reliability = sum(vm_reliability.values()) / len(vm_reliability)
        
        # Update fitness
        self.fitness = {
            "makespan": makespan,
            "cost": total_cost,
            "reliability": reliability
        }
        
        return self.fitness
    
    def crossover(self, other):
        child = Individual(self.workflow, self.cloud_env)
        
        # Single-point crossover
        crossover_point = random.randint(1, len(self.chromosome) - 1)
        task_ids = list(self.chromosome.keys())
        
        for i, task_id in enumerate(task_ids):
            if i < crossover_point:
                child.chromosome[task_id] = self.chromosome[task_id]
            else:
                child.chromosome[task_id] = other.chromosome[task_id]
        
        return child
    
    def mutate(self, mutation_rate=0.1):
        for task_id in self.chromosome:
            if random.random() < mutation_rate:
                self.chromosome[task_id] = random.choice(self.cloud_env.vms).vm_id
                
    def clone(self):
        clone = Individual(self.workflow, self.cloud_env)
        clone.chromosome = self.chromosome.copy()
        clone.fitness = self.fitness.copy()
        clone.schedule = self.schedule.copy()
        clone.completion_times = self.completion_times.copy()
        return clone

class NeuralNetworkModel:
    def __init__(self, input_size, hidden_layers=2, neurons_per_layer=32):
        self.input_size = input_size
        self.model = Sequential()
        
        # Input layer
        self.model.add(Dense(neurons_per_layer, activation='relu', input_shape=(input_size,)))
        self.model.add(Dropout(0.2))
        
        # Hidden layers
        for _ in range(hidden_layers):
            self.model.add(Dense(neurons_per_layer, activation='relu'))
            self.model.add(Dropout(0.2))
        
        # Output layer - predicting fitness values (makespan, cost, reliability)
        self.model.add(Dense(3, activation='linear'))
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='mse')
        
        # Data scalers
        self.X_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        
        # Training data
        self.X_train = None
        self.y_train = None
        
    def extract_features(self, individual):
        # Extract features from an individual's chromosome
        # Convert the chromosome representation to a feature vector
        
        task_features = []
        vm_assignment = {}
        
        for task_id, vm_id in individual.chromosome.items():
            task = individual.workflow.tasks[task_id]
            
            # Initialize VM assignment count
            if vm_id not in vm_assignment:
                vm_assignment[vm_id] = 0
            vm_assignment[vm_id] += 1
            
            # Get VM
            vm = None
            for v in individual.cloud_env.vms:
                if v.vm_id == vm_id:
                    vm = v
                    break
            
            # Features per task
            task_features.append([
                task.instructions / 1000,  # Instructions in thousands
                task.input_size,           # Input size in MB
                task.output_size,          # Output size in MB
                task.complexity,           # Task complexity
                len(task.dependencies),    # Number of dependencies
                vm.cpu_cores,              # VM CPU cores
                vm.memory,                 # VM memory in GB
                vm.cost_per_hour           # VM cost per hour
            ])
        
        # Flatten task features
        flattened_features = []
        for tf in task_features:
            flattened_features.extend(tf)
        
        # Add global features
        global_features = [
            len(individual.workflow.tasks),                     # Number of tasks
            len(individual.cloud_env.vms),                      # Number of VMs
            individual.workflow.deadline,                       # Workflow deadline
            individual.cloud_env.network_bandwidth,             # Network bandwidth
            sum(vm_assignment.values()) / len(vm_assignment)    # Average tasks per VM
        ]
        
        # Combine all features
        features = flattened_features + global_features
        
        # If the feature vector is too long, truncate or pad
        if len(features) > self.input_size:
            features = features[:self.input_size]
        elif len(features) < self.input_size:
            features.extend([0] * (self.input_size - len(features)))
        
        return np.array(features).reshape(1, -1)
    
    def collect_training_data(self, population, generations=10):
        X = []
        y = []
        
        for _ in range(generations):
            for individual in population:
                individual.calculate_fitness()
                features = self.extract_features(individual).flatten()
                fitness = [individual.fitness["makespan"], 
                          individual.fitness["cost"], 
                          individual.fitness["reliability"]]
                
                X.append(features)
                y.append(fitness)
        
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        # Scale the data
        self.X_scaler.fit(self.X_train)
        self.y_scaler.fit(self.y_train)
        
    def train(self, epochs=50, batch_size=32):
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data available. Call collect_training_data first.")
        
        # Scale the data
        X_scaled = self.X_scaler.transform(self.X_train)
        y_scaled = self.y_scaler.transform(self.y_train)
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train the model
        self.model.fit(
            X_scaled, y_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
    
    def predict_fitness(self, individual):
        features = self.extract_features(individual)
        features_scaled = self.X_scaler.transform(features)
        
        prediction_scaled = self.model.predict(features_scaled, verbose=0)
        prediction = self.y_scaler.inverse_transform(prediction_scaled)
        
        return {
            "makespan": prediction[0][0],
            "cost": prediction[0][1],
            "reliability": prediction[0][2]
        }
    
    def save_model(self, filename="nn_model"):
        self.model.save(f"{filename}.h5")
        
        # Save scalers
        with open(f"{filename}_scalers.pkl", "wb") as f:
            pickle.dump({
                "X_scaler": self.X_scaler,
                "y_scaler": self.y_scaler
            }, f)
    
    def load_model(self, filename="nn_model"):
        self.model = load_model(f"{filename}.h5")
        
        # Load scalers
        with open(f"{filename}_scalers.pkl", "rb") as f:
            scalers = pickle.load(f)
            self.X_scaler = scalers["X_scaler"]
            self.y_scaler = scalers["y_scaler"]

class NNSGA2:
    def __init__(self, workflow, cloud_env, population_size=50, nn_input_size=100):
        self.workflow = workflow
        self.cloud_env = cloud_env
        self.population_size = population_size
        self.population = []
        self.generations = 0
        self.nn_model = NeuralNetworkModel(input_size=nn_input_size)
        self.use_nn = False  # Whether to use NN for fitness prediction
        self.training_interval = 5  # Train NN every X generations
        self.nn_accuracy = 0
        self.pareto_front = []  # Current Pareto front
        
    def initialize(self):
        self.population = []
        for _ in range(self.population_size):
            individual = Individual(self.workflow, self.cloud_env)
            individual.initialize()
            self.population.append(individual)
    
    def evaluate_population(self):
        for individual in self.population:
            if self.use_nn:
                # Use NN to predict fitness
                predicted_fitness = self.nn_model.predict_fitness(individual)
                individual.fitness = predicted_fitness
            else:
                # Calculate actual fitness
                individual.calculate_fitness()
    
    def non_dominated_sort(self, population):
        # Initialize domination counters and dominated sets
        domination_count = {i: 0 for i in range(len(population))}
        dominated_set = {i: [] for i in range(len(population))}
        fronts = [[]]  # First front (Pareto front)
        
        for i in range(len(population)):
            for j in range(len(population)):
                if i == j:
                    continue
                
                ind_i = population[i]
                ind_j = population[j]
                
                # Check if i dominates j
                if (ind_i.fitness["makespan"] <= ind_j.fitness["makespan"] and
                    ind_i.fitness["cost"] <= ind_j.fitness["cost"] and
                    ind_i.fitness["reliability"] >= ind_j.fitness["reliability"] and
                    (ind_i.fitness["makespan"] < ind_j.fitness["makespan"] or
                     ind_i.fitness["cost"] < ind_j.fitness["cost"] or
                     ind_i.fitness["reliability"] > ind_j.fitness["reliability"])):
                    
                    dominated_set[i].append(j)
                
                # Check if j dominates i
                elif (ind_j.fitness["makespan"] <= ind_i.fitness["makespan"] and
                      ind_j.fitness["cost"] <= ind_i.fitness["cost"] and
                      ind_j.fitness["reliability"] >= ind_i.fitness["reliability"] and
                      (ind_j.fitness["makespan"] < ind_i.fitness["makespan"] or
                       ind_j.fitness["cost"] < ind_i.fitness["cost"] or
                       ind_j.fitness["reliability"] > ind_i.fitness["reliability"])):
                    
                    domination_count[i] += 1
            
            # If no one dominates i, it belongs to the first front
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # Identify other fronts
        front_index = 0
        while fronts[front_index]:
            next_front = []
            
            for i in fronts[front_index]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            front_index += 1
            if next_front:
                fronts.append(next_front)
        
        # Convert indices to individuals
        fronts_individuals = [[population[i] for i in front] for front in fronts]
        return fronts_individuals
    
    def crowding_distance(self, front):
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return
        
        # Initialize crowding distance
        for individual in front:
            individual.crowding_distance = 0
        
        # Calculate crowding distance for each objective
        objectives = ["makespan", "cost", "reliability"]
        for objective in objectives:
            # Sort by the objective
            if objective == "reliability":
                # For reliability, sort in descending order (higher is better)
                front.sort(key=lambda x: x.fitness[objective], reverse=True)
            else:
                # For makespan and cost, sort in ascending order (lower is better)
                front.sort(key=lambda x: x.fitness[objective])
            
            # Set infinite distance to boundary points
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Normalize the objective values
            if len(front) > 2:
                objective_range = front[-1].fitness[objective] - front[0].fitness[objective]
                
                if objective_range > 0:
                    for i in range(1, len(front) - 1):
                        # Add normalized distance to the crowding distance
                        front[i].crowding_distance += (
                            (front[i + 1].fitness[objective] - front[i - 1].fitness[objective]) /
                            objective_range
                        )
    
    def crowded_comparison(self, individual1, individual2):
        # Compare two individuals based on rank and crowding distance
        if hasattr(individual1, 'rank') and hasattr(individual2, 'rank'):
            if individual1.rank < individual2.rank:
                return individual1
            elif individual1.rank > individual2.rank:
                return individual2
        
        if hasattr(individual1, 'crowding_distance') and hasattr(individual2, 'crowding_distance'):
            if individual1.crowding_distance > individual2.crowding_distance:
                return individual1
            else:
                return individual2
        
        # Default comparison based on makespan
        if individual1.fitness["makespan"] < individual2.fitness["makespan"]:
            return individual1
        else:
            return individual2
    
    def selection(self):
        # Select parents using tournament selection
        selected = []
        
        for _ in range(self.population_size):
            # Select 2 random individuals for tournament
            tournament = random.sample(self.population, 2)
            winner = self.crowded_comparison(tournament[0], tournament[1])
            selected.append(winner)
        
        return selected
    
    def create_offspring(self, parents):
        offspring = []
        
        while len(offspring) < self.population_size:
            # Select two parents
            parent1, parent2 = random.sample(parents, 2)
            
            # Crossover
            if random.random() < 0.9:  # Crossover probability
                child = parent1.crossover(parent2)
                
                # Mutation
                child.mutate(mutation_rate=0.1)
                
                offspring.append(child)
        
        return offspring
    
    def train_nn_model(self):
        # Collect training data from the population
        self.nn_model.collect_training_data(self.population, generations=1)
        
        # Train the model
        self.nn_model.train(epochs=20, batch_size=min(32, len(self.population)))
        
        # Evaluate model accuracy
        test_individuals = random.sample(self.population, min(10, len(self.population)))
        error_sum = 0
        
        for individual in test_individuals:
            # Calculate actual fitness
            actual_fitness = individual.calculate_fitness()
            
            # Predict fitness using NN
            predicted_fitness = self.nn_model.predict_fitness(individual)
            
            # Calculate relative error for makespan and cost
            makespan_error = abs(predicted_fitness["makespan"] - actual_fitness["makespan"]) / actual_fitness["makespan"]
            cost_error = abs(predicted_fitness["cost"] - actual_fitness["cost"]) / actual_fitness["cost"]
            reliability_error = abs(predicted_fitness["reliability"] - actual_fitness["reliability"]) / max(0.01, actual_fitness["reliability"])
            
            error_sum += (makespan_error + cost_error + reliability_error) / 3
        
        self.nn_accuracy = 1 - (error_sum / len(test_individuals))
        
        # Enable NN predictions if accuracy is above threshold
        self.use_nn = self.nn_accuracy > 0.8
        
    def evolve(self, generations=100):
        # Initialize population
        self.initialize()
        
        # Main evolutionary loop
        for generation in range(generations):
            self.generations = generation + 1
            
            # Evaluate population
            self.evaluate_population()
            
            # Train NN model periodically
            if generation % self.training_interval == 0:
                self.train_nn_model()
                print(f"Generation {generation}: NN accuracy = {self.nn_accuracy:.4f}, Using NN: {self.use_nn}")
            
            # Non-dominated sorting
            fronts = self.non_dominated_sort(self.population)
            
            # Save current Pareto front
            self.pareto_front = fronts[0]
            
            # Assign ranks
            for i, front in enumerate(fronts):
                for individual in front:
                    individual.rank = i
            
            # Calculate crowding distance
            for front in fronts:
                self.crowding_distance(front)
            
            # Selection
            parents = self.selection()
            
            # Create offspring
            offspring = self.create_offspring(parents)
            
            # Evaluate offspring
            for individual in offspring:
                if self.use_nn:
                    # Use NN to predict fitness
                    predicted_fitness = self.nn_model.predict_fitness(individual)
                    individual.fitness = predicted_fitness
                else:
                    # Calculate actual fitness
                    individual.calculate_fitness()
            
            # Combine parents and offspring
            combined = self.population + offspring
            
            # Select the next generation
            self.population = []
            
            # Non-dominated sorting of combined population
            fronts = self.non_dominated_sort(combined)
            
            # Fill new population from fronts
            for front in fronts:
                if len(self.population) + len(front) <= self.population_size:
                    # Add entire front
                    self.population.extend(front)
                else:
                    # Calculate crowding distance
                    self.crowding_distance(front)
                    
                    # Sort by crowding distance
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    
                    # Add individuals until population is full
                    remaining = self.population_size - len(self.population)
                    self.population.extend(front[:remaining])
                    break
            
            # Print progress
            if generation % 10 == 0 or generation == generations - 1:
                best = min(self.population, key=lambda x: x.fitness["makespan"])
                print(f"Generation {generation}: Best makespan = {best.fitness['makespan']:.2f}, "
                      f"Cost = {best.fitness['cost']:.2f}, Reliability = {best.fitness['reliability']:.4f}")
        
        # Re-evaluate final population without using NN
        self.use_nn = False
        self.evaluate_population()
        
        # Update Pareto front
        fronts = self.non_dominated_sort(self.population)
        self.pareto_front = fronts[0]
        
        return self.pareto_front
    
    def get_best_solution(self, priority="makespan"):
        if not self.pareto_front:
            return None
        
        if priority == "makespan":
            return min(self.pareto_front, key=lambda x: x.fitness["makespan"])
        elif priority == "cost":
            return min(self.pareto_front, key=lambda x: x.fitness["cost"])
        elif priority == "reliability":
            return max(self.pareto_front, key=lambda x: x.fitness["reliability"])
        else:
            # Return a balanced solution
            normalized_scores = []
            
            # Get min and max values for each objective
            makespan_values = [ind.fitness["makespan"] for ind in self.pareto_front]
            cost_values = [ind.fitness["cost"] for ind in self.pareto_front]
            reliability_values = [ind.fitness["reliability"] for ind in self.pareto_front]
            
            makespan_min, makespan_max = min(makespan_values), max(makespan_values)
            cost_min, cost_max = min(cost_values), max(cost_values)
            reliability_min, reliability_max = min(reliability_values), max(reliability_values)
            
            # Calculate normalized scores
            for ind in self.pareto_front:
                makespan_score = (ind.fitness["makespan"] - makespan_min) / max(1e-6, makespan_max - makespan_min)
                cost_score = (ind.fitness["cost"] - cost_min) / max(1e-6, cost_max - cost_min)
                reliability_score = 1 - ((ind.fitness["reliability"] - reliability_min) / max(1e-6, reliability_max - reliability_min))
                
                # Lower is better
                total_score = makespan_score + cost_score + reliability_score
                normalized_scores.append((ind, total_score))
            
            # Return individual with lowest total score
            return min(normalized_scores, key=lambda x: x[1])[0]
    
    def plot_pareto_front(self):
        if not self.pareto_front:
            print("No Pareto front to plot")
            return
        
        # Extract fitness values
        makespan = [ind.fitness["makespan"] for ind in self.pareto_front]
        cost = [ind.fitness["cost"] for ind in self.pareto_front]
        reliability = [ind.fitness["reliability"] for ind in self.pareto_front]
        
        # Create figure
        fig = plt.figure(figsize=(15, 5))
        
        # Plot makespan vs cost
        ax1 = fig.add_subplot(131)
        scatter1 = ax1.scatter(makespan, cost, c=reliability, cmap='viridis')
        ax1.set_xlabel('Makespan')
        ax1.set_ylabel('Cost')
        ax1.set_title('Makespan vs Cost')
        fig.colorbar(scatter1, ax=ax1, label='Reliability')
        
        # Plot makespan vs reliability
        ax2 = fig.add_subplot(132)
        scatter2 = ax2.scatter(makespan, reliability, c=cost, cmap='viridis')
        ax2.set_xlabel('Makespan')
        ax2.set_ylabel('Reliability')
        ax2.set_title('Makespan vs Reliability')
        fig.colorbar(scatter2, ax=ax2, label='Cost')
        
        # Plot cost vs reliability
        ax3 = fig.add_subplot(133)
        scatter3 = ax3.scatter(cost, reliability, c=makespan, cmap='viridis')
        ax3.set_xlabel('Cost')
        ax3.set_ylabel('Reliability')
        ax3.set_title('Cost vs Reliability')
        fig.colorbar(scatter3, ax=ax3, label='Makespan')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig('pareto_front.png')
        plt.close()

def create_synthetic_workflow(num_tasks=20, max_dependencies=3):
    workflow = Workflow(workflow_id=1)
    
    # Create tasks
    for i in range(1, num_tasks + 1):
        instructions = random.randint(1000, 10000)  # Instructions in millions
        input_size = random.randint(10, 1000)       # Input size in MB
        output_size = random.randint(10, 1000)      # Output size in MB
        complexity = random.randint(1, 10)          # Complexity on a scale of 1-10
        
        task = Task(i, instructions, input_size, output_size, complexity)
        workflow.add_task(task)
    
    # Create dependencies (ensuring a DAG)
    for i in range(2, num_tasks + 1):
        # Each task can depend on any previous task
        num_dependencies = random.randint(0, min(max_dependencies, i - 1))
        dependencies = random.sample(range(1, i), num_dependencies)
        
        for dep in dependencies:
            workflow.tasks[i].add_dependency(dep)
    
    # Set a reasonable deadline
    # First, calculate a rough estimate of the minimum completion time
    min_time = sum([task.instructions / 1000 for task in workflow.tasks.values()])
    workflow.set_deadline(min_time * 1.5)  # 50% slack
    
    return workflow

def create_cloud_environment(num_vms=5):
    cloud_env = CloudEnvironment()
    
    # VM configurations - (cpu_cores, memory, cost_per_hour)
    vm_configs = [
        (2, 4, 0.10),   # Small
        (4, 8, 0.20),   # Medium
        (8, 16, 0.40),  # Large
        (16, 32, 0.80), # XLarge
        (32, 64, 1.60)  # 2XLarge
    ]
    
    # Create VMs
    for i in range(1, num_vms + 1):
        # Select a random VM configuration
        config = random.choice(vm_configs)
        
        vm = VirtualMachine(
            vm_id=i,
            cpu_cores=config[0],
            memory=config[1],
            cost_per_hour=config[2]
        )
        
        cloud_env.add_vm(vm)
    
    return cloud_env

def save_results(pareto_front, filename="results.csv"):
    results = []
    for i, individual in enumerate(pareto_front):
        results.append({
            "solution_id": i,
            "makespan": individual.fitness["makespan"],
            "cost": individual.fitness["cost"],
            "reliability": individual.fitness["reliability"]
        })
        
        # Add task assignments
        for task_id, vm_id in individual.chromosome.items():
            results[-1][f"task_{task_id}_vm"] = vm_id
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    
    print(f"Results saved to {filename}")

def visualize_schedule(individual, filename="schedule.png"):
    # Create Gantt chart of the schedule
    vm_map = {vm.vm_id: vm for vm in individual.cloud_env.vms}
    
    # Prepare data for Gantt chart
    tasks = []
    for task_id, start_time in individual.schedule.items():
        task = individual.workflow.tasks[task_id]
        vm_id = individual.chromosome[task_id]
        vm = vm_map[vm_id]
        
        duration = vm.calculate_task_execution_time(task)
        
        tasks.append({
            "Task": f"Task {task_id}",
            "Start": start_time,
            "Finish": start_time + duration,
            "Resource": f"VM {vm_id}"
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(tasks)
    
    # Sort by start time
    df = df.sort_values("Start")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Unique resources
    resources = df["Resource"].unique()
    
    # Colors for resources
    colors = plt.cm.tab10(np.linspace(0, 1, len(resources)))
    resource_colors = {resource: color for resource, color in zip(resources, colors)}
    
    # Plot tasks
    for i, task in df.iterrows():
        ax.barh(task["Task"], 
                task["Finish"] - task["Start"], 
                left=task["Start"], 
                color=resource_colors[task["Resource"]],
                alpha=0.8)
        
        # Add task ID text
        ax.text(task["Start"] + (task["Finish"] - task["Start"]) / 2,
                task["Task"],
                task["Task"],
                ha="center",
                va="center",
                color="black")
    
    # Add legend
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color in resource_colors.values()]
    ax.legend(legend_patches, resource_colors.keys(), loc="upper right")
    
    # Set labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Tasks")
    ax.set_title("Task Schedule Gantt Chart")
    
    # Set y-axis limits
    ax.set_ylim(-0.5, len(df["Task"].unique()) - 0.5)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Schedule visualization saved to {filename}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create workflow and cloud environment
    print("Creating workflow and cloud environment...")
    workflow = create_synthetic_workflow(num_tasks=20, max_dependencies=3)
    cloud_env = create_cloud_environment(num_vms=5)
    
    # Print workflow and cloud environment details
    print(f"Workflow {workflow.workflow_id} created with {len(workflow.tasks)} tasks")
    print(f"Cloud environment created with {len(cloud_env.vms)} VMs")
    
    # Print task details
    print("\nTask details:")
    for task_id, task in workflow.tasks.items():
        deps = ', '.join(map(str, task.dependencies)) if task.dependencies else 'None'
        print(f"Task {task_id}: instructions={task.instructions}, input={task.input_size}MB, "
              f"output={task.output_size}MB, complexity={task.complexity}, dependencies={deps}")
    
    # Print VM details
    print("\nVM details:")
    for vm in cloud_env.vms:
        print(f"VM {vm.vm_id}: {vm.cpu_cores} cores, {vm.memory}GB RAM, ${vm.cost_per_hour}/hour")
    
    # Create NNSGA2 optimizer
    print("\nInitializing NNSGA2 optimizer...")
    nnsga2 = NNSGA2(workflow, cloud_env, population_size=50, nn_input_size=100)
    
    # Run evolution
    print("\nStarting evolution...")
    start_time = time.time()
    pareto_front = nnsga2.evolve(generations=100)
    elapsed_time = time.time() - start_time
    print(f"\nEvolution completed in {elapsed_time:.2f} seconds")
    
    # Get best solutions
    print("\nBest solutions from Pareto front:")
    
    best_makespan = nnsga2.get_best_solution(priority="makespan")
    best_cost = nnsga2.get_best_solution(priority="cost")
    best_reliability = nnsga2.get_best_solution(priority="reliability")
    balanced = nnsga2.get_best_solution(priority="balanced")
    
    print(f"Best makespan solution: {best_makespan.fitness}")
    print(f"Best cost solution: {best_cost.fitness}")
    print(f"Best reliability solution: {best_reliability.fitness}")
    print(f"Balanced solution: {balanced.fitness}")
    
    # Save neural network model
    print("\nSaving neural network model...")
    nnsga2.nn_model.save_model("nn_workflow_scheduler")
    
    # Plot Pareto front
    print("\nPlotting Pareto front...")
    nnsga2.plot_pareto_front()
    
    # Save results
    print("\nSaving results...")
    save_results(pareto_front, "pareto_front_results.csv")
    
    # Visualize best schedule
    print("\nVisualizing best schedule...")
    visualize_schedule(balanced, "best_schedule.png")
    
    # Print final solution details
    print("\nFinal solution details:")
    for task_id, vm_id in balanced.chromosome.items():
        vm = None
        for v in cloud_env.vms:
            if v.vm_id == vm_id:
                vm = v
                break
        
        task = workflow.tasks[task_id]
        start_time = balanced.schedule[task_id]
        exec_time = vm.calculate_task_execution_time(task)
        
        print(f"Task {task_id} assigned to VM {vm_id}, starts at {start_time:.2f}, "
              f"completes at {start_time + exec_time:.2f}, duration {exec_time:.2f}")
    
    print("\nExecution complete!")

if __name__ == "__main__":
    main()