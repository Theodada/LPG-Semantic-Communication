SPACEX SEMANTIC COMMUNICATIONS LIVE DEMO

Purpose
This package gives a reviewer a click to run demo from the larger LPG semantic communication work. It runs two simulations:

S3 Anchor Dynamics and Capacity Evolution
Shows enrichment, misalignment, clarification, and capacity super recovery.

S5 6G V2X Semantic Communication Under URLLC Constraints
Shows that the LPG semantic proxy reaches the CAR 0.95 safety target at lower SNR than classical and DL semantic baselines.

Windows
Double click RUN_SPACEX_DEMO_WINDOWS.bat

Mac or Linux
Open Terminal in this folder and run:
./RUN_SPACEX_DEMO_MAC_LINUX.sh

Direct Python command
python spacex_semantic_demo.py --animate

Requirements
Python 3.9 or newer
numpy
matplotlib

Install requirements when needed
python -m pip install numpy matplotlib

Generated outputs
The script creates an output folder containing:
data CSV files
plot PNG files
report/SpaceX_semantic_communications_demo_report.html

Engineering caveat
This is a deterministic proxy demo, not the full CARLA, SUMO, ns 3, Sionna, transformer and dataset validation stack. It is designed to make the theory runnable and inspectable during a short technical review.
