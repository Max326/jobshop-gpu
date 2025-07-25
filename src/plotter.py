import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_makespans():
    """
    Loads makespan data from CSV files and generates a comparative plot.

    This function reads training and validation makespan data, then plots both
    on a single graph to visualize performance over iterations.
    """
    try:
        # --- IMPORTANT ---
        # Define the path to the directory containing your CSV files.
        # This path should be relative to your project's root folder.
        data_directory = "data/RESULTS/rnd_JT(10)_J(15)_M(5)_JO(5-10)_O(20)_OM(1-3)/"

        # Construct the full paths for the CSV files
        val_file_path = os.path.join(data_directory, 'best_val_makespans.csv')
        train_file_path = os.path.join(data_directory, 'best_train_makespans.csv')

        # --- For Debugging ---
        # You can print the paths to see what the script is trying to open
        # print(f"Trying to open validation file: {os.path.abspath(val_file_path)}")
        # print(f"Trying to open training file: {os.path.abspath(train_file_path)}")
        
        # Load the datasets from the constructed file paths
        val_makespans = pd.read_csv(val_file_path)
        train_makespans = pd.read_csv(train_file_path)

        # Create a plot with a specified figure size
        plt.figure(figsize=(12, 7))

        # Plot the validation makespan data
        plt.plot(val_makespans['iteration'], val_makespans['val_avg_makespan'], label='Validation Avg Makespan')

        # Plot the training makespan data
        plt.plot(train_makespans['iteration'], train_makespans['best_train_avg_makespan'], label='Training Best Avg Makespan')

        # Add labels and title for clarity
        plt.xlabel("Iteration")
        plt.ylabel("Makespan")
        plt.title("Comparison of Training and Validation Makespans")
        
        # Add a legend to identify the lines
        plt.legend()
        plt.grid(True)
        
        # Display the plot
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: {e}.")
        print("Please ensure the 'data_directory' variable in the script is set correctly.")
        # For more help, you can see the current working directory by uncommenting the line below
        # print(f"Current Working Directory: {os.getcwd()}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Execute the function to generate the plot
plot_makespans()
