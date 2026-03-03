SecGrad-XAI: Secure and Accurate Explainable AI
This repository contains the official implementation for "SecGrad-XAI: Secure and Accurate Explainable AI 
with Secure Multi-Party Image Classification", published in the proceedings of the EICC'26 conference.

Figure 1: Overview of the SecGrad-XAI framework for gradient-based explanations during SNNI.
[SecGrad-XAI.pdf](https://github.com/user-attachments/files/25721791/SecGrad-XAI.pdf)

SecGrad-XAI is a framework designed to enable gradient-based explanations (XAI) during Secure Neural Network Inference (SNNI). 
It utilizes the CrypTen framework, leveraging secret-sharing protocols to ensure privacy during both the inference and explanation phases.

Repository Structure
This repository is organized as a monorepo for version control, though the components are intended to run on separate physical machines:

client/: Source code to be deployed on the Client machine.

server/: Source code to be deployed on the Server machine.


Inside each machine's code base (we suggest naming the folders "xai" on each machine instead of the given "client"/"server".), there should be
two additional folders:

models/: Directory for trained neural network models (.pth files).
data/: Directory for inference datasets (only the test datasets are necessary post-training).

Deployment & Setup
1. File Distribution
Because this is an MPC-based protocol, the code must be distributed across two distinct environments:

On the Client Machine: Copy the contents of the client/ folder into your working directory.

On the Server Machine: Copy the contents of the server/ folder into your working directory.

Both: Ensure the models/ and data/ folders are present on both machines.

2. Environment Initialization
Both machines should use a consistent environment (e.g., myenv). Navigate to your working directory on each machine and run:

Bash
pip install -r requirements.txt
3. Network & Interface Configuration
The machines must be synchronized via two shell scripts before execution:

common.sh: Both machines must specify the Server's IP address and use an identical Gloo socket name (GLOO_SOCKET_IFNAME) to establish the CrypTen backend.

throttle.sh: Both machines must set the DEV= variable to the correct network interface used for communication.

Execution Flow
The framework requires a specific "Server-First" startup sequence.

Step 1: Parameter Selection
Ensure the following parameters are identical in the run.sh files on both machines:

Neural Network Architecture

Dataset

Mode (plaintext or secure)

XAI Method

Note: The client's run.sh contains additional parameters. The server's version is intentionally limited to ensure specific inference decisions and parameters remain confidential from the server.

Step 2: Run the Server
On the server machine:

Bash
./run.sh
Step 3: Run the Client
On the client machine:

Bash
./run.sh
Citation
If you use this code in your research, please cite our EICC'26 paper
