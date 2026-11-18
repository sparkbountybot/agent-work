#!/bin/bash
# task_runner.sh — Automated task runner for agent-work helper machines
# Run this every 30-60 minutes on helper machines

set -e

# Configuration
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS_DIR="$REPO_DIR/scripts"
TASKS_DIR="$REPO_DIR/tasks"
INPROGRESS_DIR="$REPO_DIR/in-progress"
RESULTS_DIR="$REPO_DIR/results"
TEMPLATES_DIR="$REPO_DIR/templates"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Check if required directories exist
check_dirs() {
    for dir in "$TASKS_DIR" "$INPROGRESS_DIR" "$RESULTS_DIR"; do
        if [ ! -d "$dir" ]; then
            log_error "Directory not found: $dir"
            log_info "Creating missing directory..."
            mkdir -p "$dir"
        fi
    done
}

# Pull latest changes from remote
pull_updates() {
    log_info "Pulling latest changes from remote..."
    if git pull --rebase origin main 2>&1; then
        log_success "Updated successfully"
        return 0
    else
        log_error "Failed to pull updates. Try manually: git pull --rebase origin main"
        return 1
    fi
}

# Find available tasks
find_available_tasks() {
    log_info "Looking for available tasks..."
    
    # Get list of available task files
    local tasks=$(find "$TASKS_DIR" -name "*.md" -type f 2>/dev/null)
    
    if [ -z "$tasks" ]; then
        log_info "No tasks available"
        return 0
    fi
    
    # Filter out tasks that are in-progress
    local available_count=0
    for task in $tasks; do
        local task_id=$(basename "$task" .md)
        if [ ! -f "$INPROGRESS_DIR/${task_id}_lock.json" ]; then
            available_count=$((available_count + 1))
            # Display task info
            local title=$(grep -m1 '^#' "$task" | head -c 50)
            local priority=$(grep -m1 '^priority:' "$task" | awk '{print $2}')
            log_info "  Task $task_id: $title [Priority: $priority]"
        else
            log_info "  Task $task_id: Skipped (already in progress)"
        fi
    done
    
    if [ $available_count -eq 0 ]; then
        log_info "No tasks available right now"
        return 0
    fi
    
    log_success "Found $available_count available task(s)"
}

# Claim a task
claim_task() {
    local task_file="$1"
    
    if [ -z "$task_file" ]; then
        # Find first available task
        task_file=$(find "$TASKS_DIR" -name "*.md" -type f 2>/dev/null | head -1)
    fi
    
    if [ -z "$task_file" ]; then
        log_error "No task to claim"
        return 1
    fi
    
    local task_id=$(basename "$task_file" .md)
    
    # Check if already claimed
    if [ -f "$INPROGRESS_DIR/${task_id}_lock.json" ]; then
        log_warning "Task $task_id already claimed by $(cat "$INPROGRESS_DIR/${task_id}_lock.json")"
        return 1
    fi
    
    # Claim the task
    log_info "Claiming task $task_id..."
    
    # Create lock file with machine info
    local machine_name=$(hostname)
    local timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    
    cat > "$INPROGRESS_DIR/${task_id}_lock.json" << EOF
{
  "claimed_by": "$(whoami)",
  "machine": "$machine_name",
  "timestamp": "$timestamp",
  "pid": $$
}
EOF
    
    log_success "Task $task_id claimed by $machine_name"
    
    # Move task to in-progress (create a copy, don't delete from tasks)
    cp "$task_file" "$INPROGRESS_DIR/${task_id}.md"
    
    return 0
}

# Submit a result
submit_result() {
    local result_file="$1"
    
    if [ -z "$result_file" ]; then
        log_info "Usage: $0 submit /path/to/result.json"
        return 1
    fi
    
    if [ ! -f "$result_file" ]; then
        log_error "Result file not found: $result_file"
        return 1
    fi
    
    # Read task_id from result file
    local task_id=$(python3 -c "import json; print(json.load(open('$result_file'))['task_id'])" 2>/dev/null)
    
    if [ -z "$task_id" ]; then
        log_error "Invalid result file format (missing task_id)"
        return 1
    fi
    
    log_info "Submitting result for task $task_id..."
    
    # Copy result to results directory
    cp "$result_file" "$RESULTS_DIR/${task_id}_result.json"
    
    # Remove the lock file (task is complete)
    rm -f "$INPROGRESS_DIR/${task_id}_lock.json"
    
    log_success "Result submitted for task $task_id"
    
    # Commit and push
    log_info "Committing result..."
    git add "$RESULTS_DIR/" "$INPROGRESS_DIR/" 2>/dev/null
    git commit -m "result: submit result for task $task_id" 2>&1 || true
    git push origin main 2>&1 || log_warning "Push failed (may need manual intervention)"
    
    return 0
}

# Show status
show_status() {
    log_info "Task Status:"
    echo ""
    
    local pending=$(find "$TASKS_DIR" -name "*.md" -type f 2>/dev/null | wc -l)
    local in_progress=$(find "$INPROGRESS_DIR" -name "*.md" -type f 2>/dev/null | wc -l)
    local completed=$(find "$RESULTS_DIR" -name "*.json" -type f 2>/dev/null | wc -l)
    
    echo -e "  ${GREEN}Pending:${NC} $pending"
    echo -e "  ${YELLOW}In Progress:${NC} $in_progress"
    echo -e "  ${BLUE}Completed:${NC} $completed"
    
    echo ""
    
    # Show in-progress details
    if [ "$in_progress" -gt 0 ]; then
        log_info "In Progress Tasks:"
        for lock in "$INPROGRESS_DIR"/*_lock.json; do
            if [ -f "$lock" ]; then
                local task_id=$(basename "$lock" _lock.json)
                local by=$(python3 -c "import json; print(json.load(open('$lock'))['claimed_by'])" 2>/dev/null || echo "unknown")
                log_info "  $task_id → claimed by $by"
            fi
        done
        echo ""
    fi
    
    # Show recent results
    if [ "$completed" -gt 0 ]; then
        log_info "Recent Results:"
        for result in "$RESULTS_DIR"/*_result.json; do
            if [ -f "$result" ]; then
                local task_id=$(basename "$result" _result.json)
                local by=$(python3 -c "import json; print(json.load(open('$result'))['completed_by'])" 2>/dev/null || echo "unknown")
                log_info "  $task_id → submitted by $by"
            fi
        done
    fi
}

# Main execution
main() {
    log_info "Agent Work Task Runner v1.0"
    log_info "Repo: $REPO_DIR"
    echo ""
    
    # Check directories
    check_dirs
    
    # Check command
    case "${1:-check}" in
        check|default)
            # Default mode: check for tasks and run available work
            pull_updates || exit 1
            find_available_tasks
            claim_task  # Try to claim a task
            ;;
        pull)
            # Just pull updates
            pull_updates || exit 1
            ;;
        claim)
            # Claim first available task
            find_available_tasks
            claim_task
            ;;
        submit)
            # Submit a result
            submit_result "$2"
            ;;
        status)
            show_status
            ;;
        help)
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  check      Check for tasks and claim one (default)"
            echo "  pull       Pull latest changes"
            echo "  claim      Claim first available task"
            echo "  submit     Submit a result (pass file path as argument)"
            echo "  status     Show current status"
            echo "  help       Show this help message"
            ;;
        *)
            log_error "Unknown command: $1"
            echo "Run '$0 help' for usage information"
            exit 1
            ;;
    esac
    
    echo ""
    log_success "Task runner completed"
}

# Run main function
main "$@"
