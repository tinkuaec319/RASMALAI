#!/bin/bash

# Configuration
SESSION_NAME="nomod_run"
MAX_CONCURRENT=4  # Total concurrent processes allowed

# Ordered list of scripts (calculation first, then others)
SCRIPTS=(
    "NoMod%_calculation.py"
    "NoMod_60%_secret_recovery_results.py"
    "NoMod_65%_secret_recovery_results.py"
    "NoMod_70%_secret_recovery_results.py"
    "NoMod_75%_secret_recovery_results.py"
    "NoMod_80%_secret_recovery_results.py"
    "NoMod_85%_secret_recovery_results.py"
    "NoMod_90%_secret_recovery_results.py"
    "NoMod_90%_secret_recovery_results_with_model_names.py"
)

# Create new tmux session
tmux new-session -d -s "$SESSION_NAME" -n "control_panel"

# Function to count running processes
count_running() {
    tmux list-windows -t "$SESSION_NAME" | grep -cvE "control_panel|^[0-9]+:"
}

# Start initial batch
for ((i=0; i<$MAX_CONCURRENT && i<${#SCRIPTS[@]}; i++)); do
    script="${SCRIPTS[$i]}"
    tmux new-window -t "$SESSION_NAME" -n "${script%.*}" "python3 '$script'"
done

# Process remaining scripts
for ((i=$MAX_CONCURRENT; i<${#SCRIPTS[@]}; i++)); do
    while true; do
        current_count=$(count_running)
        [ $current_count -lt $MAX_CONCURRENT ] && break
        sleep 10
    done
    
    script="${SCRIPTS[$i]}"
    tmux new-window -t "$SESSION_NAME" -n "${script%.*}" "python3 '$script'"
done

# Monitoring instructions
echo -e "\n\033[1;36mProcesses started in tmux session: $SESSION_NAME\033[0m"
echo -e "Use these commands to monitor:"
echo -e "  tmux attach -t $SESSION_NAME  # Attach to session"
echo -e "  Ctrl+B then D                 # Detach from session"
echo -e "  tmux list-windows -t $SESSION_NAME  # List all processes"
