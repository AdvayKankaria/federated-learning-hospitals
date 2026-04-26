#!/bin/bash
# ===========================================
# Hospital Federated Learning Training Script
# ===========================================

set -e

echo "🏥 Hospital Federated Learning"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="${CONFIG_FILE:-config/config.yaml}"
DATA_DIR="${DATA_DIR:-./data/rsna-pneumonia}"
NUM_ROUNDS="${NUM_ROUNDS:-50}"
NUM_HOSPITALS="${NUM_HOSPITALS:-5}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --download)
            DOWNLOAD_DATA=true
            shift
            ;;
        --partition)
            PARTITION_DATA=true
            shift
            ;;
        --train)
            RUN_TRAINING=true
            shift
            ;;
        --explain)
            RUN_EXPLAIN=true
            shift
            ;;
        --all)
            DOWNLOAD_DATA=true
            PARTITION_DATA=true
            RUN_TRAINING=true
            RUN_EXPLAIN=true
            shift
            ;;
        --rounds)
            NUM_ROUNDS="$2"
            shift 2
            ;;
        --hospitals)
            NUM_HOSPITALS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Step 1: Download data
if [ "$DOWNLOAD_DATA" = true ]; then
    echo -e "${GREEN}📥 Step 1: Downloading RSNA Pneumonia Detection dataset...${NC}"
    python train.py --download-data --config $CONFIG_FILE
    echo ""
fi

# Step 2: Create partitions
if [ "$PARTITION_DATA" = true ]; then
    echo -e "${GREEN}📊 Step 2: Creating Non-IID data partitions for $NUM_HOSPITALS hospitals...${NC}"
    python train.py --partition-data --config $CONFIG_FILE --num-hospitals $NUM_HOSPITALS
    echo ""
fi

# Step 3: Run training
if [ "$RUN_TRAINING" = true ]; then
    echo -e "${GREEN}🚀 Step 3: Starting Federated Learning training...${NC}"
    echo -e "${YELLOW}   - Hospitals: $NUM_HOSPITALS${NC}"
    echo -e "${YELLOW}   - Rounds: $NUM_ROUNDS${NC}"
    echo -e "${YELLOW}   - Config: $CONFIG_FILE${NC}"
    echo ""
    
    python train.py --train --config $CONFIG_FILE \
        --num-rounds $NUM_ROUNDS \
        --num-hospitals $NUM_HOSPITALS
    echo ""
fi

# Step 4: Generate explanations
if [ "$RUN_EXPLAIN" = true ]; then
    echo -e "${GREEN}🔍 Step 4: Generating Grad-CAM explanations...${NC}"
    python train.py --explain --model-path checkpoints/best_model.pt
    echo ""
fi

echo -e "${GREEN}✅ Done!${NC}"
echo ""
echo "Next steps:"
echo "  - View metrics in ./metrics/"
echo "  - View explanations in ./explanations/"
echo "  - Start dashboard: cd dashboard/backend && uvicorn main:app --reload"

