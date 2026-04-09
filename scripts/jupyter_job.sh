#!/bin/bash
#SBATCH --job-name=jupyter-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --output=/home/kaushim07/photometry_mcmc_env/logs/jupyter_%j.out

# 1. FORCE move to your project directory
cd /home/kaushim07/photometry_mcmc_env/

# 2. Set up the port and info
node=$(hostname -s)
user=$(whoami)
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "========================================================="
echo "TUNNEL COMMAND: ssh -N -L 8888:${node}:${port} ${user}@login.curta.zedat.fu-berlin.de"
echo "========================================================="

module add JupyterLab/4.0.5-GCCcore-12.3.0
# Starting with no-token and no-browser for pure ease
jupyter-lab --no-browser --port=${port} --ip=${node} --ServerApp.token='' --ServerApp.password=''
