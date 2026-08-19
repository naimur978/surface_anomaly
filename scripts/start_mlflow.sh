#!/bin/bash
# Start MLflow UI with automatic process management

PORT=${1:-5000}

echo "==========================================="
echo "MLflow Server Startup"
echo "==========================================="
echo ""

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "Port $PORT is already in use."
    echo ""
    echo "Options:"
    echo "  1. Kill existing process and restart"
    echo "  2. Use different port"
    echo "  3. Cancel"
    echo ""
    read -p "Enter choice (1-3): " choice

    case $choice in
        1)
            echo "Killing process on port $PORT..."
            lsof -ti:$PORT | xargs kill -9
            sleep 2
            echo "Process killed. Starting MLflow..."
            ;;
        2)
            read -p "Enter new port (default 5001): " NEW_PORT
            PORT=${NEW_PORT:-5001}
            echo "Using port $PORT instead..."
            ;;
        3)
            echo "Cancelled."
            exit 0
            ;;
        *)
            echo "Invalid choice."
            exit 1
            ;;
    esac
fi

echo ""
echo "Starting MLflow on port $PORT..."
echo "==========================================="
echo ""

mlflow ui --backend-store-uri ./mlruns --port $PORT

echo ""
echo "==========================================="
echo "MLflow stopped"
echo "==========================================="
