#!/bin/bash
#SBATCH --job-name=jupyter-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --time=09:00:00
#SBATCH --output=/home/kaushim07/photometry_mcmc_env/logs/jupyter_%j.out

# 1. Navigate to project directory
cd /home/kaushim07/photometry_mcmc_env/

# 2. Initialize and Activate your specific environment
# We use the full path to the conda init script to ensure 'conda activate' works
source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

# 3. VERIFICATION (Look for this in your log file)
echo "--- ENVIRONMENT CHECK ---"
echo "Node: $(hostname -s)"
echo "Python Path: $(which python)"
echo "Python Version: $(python --version)"
echo "--------------------------"

# 4. Set up the dynamic port
node=$(hostname -s)
user=$(whoami)
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "========================================================="
echo "TUNNEL COMMAND: ssh -N -L 8888:${node}:${port} ${user}@login.curta.zedat.fu-berlin.de"
echo "========================================================="

# 5. Launch Jupyter Lab from WITHIN the environment
# We use $(which jupyter-lab) to ensure we aren't using a system version
$(which jupyter-lab) --no-browser --port=${port} --ip=${node} --ServerApp.token='' --ServerApp.password=''