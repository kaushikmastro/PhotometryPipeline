import os
from pathlib import Path

kernel_dir = Path("/scratch/kaushim07/vesta_data/spice_kernels")
tm_file = kernel_dir / "dawn_dynamic.tm"

# The foundational kernels that live in the root directory
core_kernels = [
    "naif0012.tls",
    "DAWN_203_SCLKSCET.00091.tsc",
    "pck00010.tpc",
    "dawn_ceres_v00.tf",
    "dawn_v11.tf",
    "dawn_vesta_v00.tf",
    "dawn_fc_v02.ti",
    "dawn_grand_v00.ti",
    "dawn_struct_v00.ti",
    "dawn_vir_v05.ti"
]

# Dynamically scrape the subdirectories you created
spk_files = sorted([f.name for f in (kernel_dir / "spk").glob("*.bsp")])
ck_files = sorted([f.name for f in (kernel_dir / "ck").glob("*.bc")])

with open(tm_file, "w") as f:
    f.write("KPL/MK\n\n\\begindata\n\n")
    f.write("PATH_VALUES = ( '/scratch/kaushim07/vesta_data/spice_kernels' )\n")
    f.write("PATH_SYMBOLS = ( 'SPICE' )\n\n")
    f.write("KERNELS_TO_LOAD = (\n")
    
    # 1. Load core kernels
    for k in core_kernels:
        f.write(f"  '$SPICE/{k}',\n")
        
    # 2. Load Trajectory files from spk/
    for k in spk_files:
        f.write(f"  '$SPICE/spk/{k}',\n")
        
    # 3. Load Spacecraft orientation files from ck/
    for k in ck_files:
        f.write(f"  '$SPICE/ck/{k}',\n")
        
    # 4. Load the DSK from the root directory
    f.write("  '$SPICE/vesta_gaskell_256.bds'\n")
    
    f.write(")\n\n\\begintext\n")
    f.write("Cleanly generated metakernel with correct subdirectories and DSK.\n")

print(f"Successfully rebuilt {tm_file.name}!")
