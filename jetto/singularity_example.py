import jetto_tools.template
import jetto_tools.config
import jetto_tools.job
from datetime import datetime
from dotenv import load_dotenv
from os import getenv
import subprocess

load_dotenv()
JETTO_TEMPLATE = getenv("JETTO_TEMPLATE")
OUTPUT_DIR = getenv("OUTPUT_DIR")
JINTRAC_IMAGE = getenv("JINTRAC_IMAGE")

run_name = "test" # datetime.now().strftime("%Y-%d-%m_%H%M_%S")

# Load JETTO run template
print(f"Loading JETTO template from {JETTO_TEMPLATE}...")
template = jetto_tools.template.from_directory(JETTO_TEMPLATE)
print("Done.")

# Generate JETTO config from template
print("Generating new config from template...")
config = jetto_tools.config.RunConfig(template)
# You can modify the config here
# e.g. config["PARAM"] = 1
# ...
config.binary = "v220922"
config.userid = "sim"
# Save config
config.export(f"{OUTPUT_DIR}/{run_name}")
print(f"Config saved to {OUTPUT_DIR}/{run_name}.")

# Run JETTO container
# This implements the command:
# singularity exec --cleanenv -B /tmp -B $OUTPUT_DIR/$RUN_NAME:/jetto/runs/$RUN_NAME $JINTRAC_IMAGE rjettov -x64 -S -p0 -n1 $RUN_NAME build docker
print(f"Launching Singularity container, name={run_name}...")
process = subprocess.run(["singularity",
    "exec",
    "--cleanenv",  # Run in a clean environment (no env variables etc)
    "--bind", "/tmp",
    "--bind", f"{OUTPUT_DIR}/{run_name}:/jetto/runs/{run_name}",  # Bind the output directory to the container's jetto run directory
    JINTRAC_IMAGE,  # Container image
    "rjettov", "-x64", "-S", "-p0", "-n1", run_name, "build", "docker",  # Command to execute in container
])
print(f"Run complete. Output saved to {OUTPUT_DIR}/{run_name}")
