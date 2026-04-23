 ISNN Assignment
Implementation of Input Specific Neural Networks (ISNN) from the paper Input Specific Neural Networks by Jadoon, Seidl, Jones, and Fuhg (2025).

What is this
This notebook implements ISNN-1 and ISNN-2 architectures in two ways — once using PyTorch and once manually using only NumPy with hand-written backpropagation. Both are trained and tested on two toy datasets from Section 3.1 of the paper.

Files
FileDescriptionISNN_Assignment.ipynbMain notebook with all codeX_train_toy1.npyTraining inputs for Toy Problem 1y_train_toy1.npyTraining labels for Toy Problem 1X_test_toy1.npyTest inputs for Toy Problem 1y_test_toy1.npyTest labels for Toy Problem 1X_train_toy2.npyTraining inputs for Toy Problem 2y_train_toy2.npyTraining labels for Toy Problem 2X_test_toy2.npyTest inputs for Toy Problem 2y_test_toy2.npyTest labels for Toy Problem 2

Requirements
numpy
matplotlib
torch
scipy
Install with:
pip install numpy matplotlib torch scipy

How to run
Open ISNN_Assignment.ipynb in Jupyter and run cells from top to bottom. Each section is clearly labeled.

Notebook Structure

Imports and setup
Dataset generation — two toy functions sampled using Latin Hypercube Sampling
PyTorch implementation — ISNN-1, ISNN-2, and FFNN baseline with autograd
Manual NumPy implementation — same models with hand-written backpropagation, no autograd
Loss plots — training and test loss curves for all models (Figure 3 and Figure 5 style)
Behavior plots — model predictions vs true function including extrapolation region (Figure 4 and Figure 6 style)
Results summary — final test MSE for all models and both datasets


Notes

Each model is trained with 10 different random initializations. Plots show mean and standard deviation across all runs.
Structural constraints (non-negative weights) are enforced after every gradient step.
The Adam optimizer is coded from scratch in the manual NumPy section.
Training is set to 5000 epochs by default. Increase to 30000 for results closer to the paper.
