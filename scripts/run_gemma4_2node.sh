#!/usr/bin/env bash
# ============================================================
# Gemma4-31B 2-node distributed inference — clean deployment
#
# Usage:
#   Node 1 (DHT + server blocks 0:30):
#     bash scripts/run_gemma4_2node.sh server1 <node2_ip>
#
#   Node 2 (server blocks 30:60 + client test):
#     bash scripts/run_gemma4_2node.sh server2 <node1_ip> <dht_peer_addr>
#
#   Client only (run from any node with network access):
#     bash scripts/run_gemma4_2node.sh client <dht_peer_addr>
#
# Prerequisites:
#   - Python 3.11 venv at ~/bloombee-env-311 with torch, transformers>=5.5
#   - Model google/gemma-4-31b-it downloaded via:
#       huggingface-cli download google/gemma-4-31b-it
#   - BloomBee cloned to ~/BloomBee (this repo)
# ============================================================
set -euo pipefail

MODEL="google/gemma-4-31b-it"
DHT_PORT=31340
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export PYTHONPATH="$REPO_DIR/src"
export BLOOMBEE_IGNORE_DEPENDENCY_VERSION=1

activate_env() {
    source ~/bloombee-env-311/bin/activate
}

# ---- server1: start DHT + serve blocks 0:30 ----
cmd_server1() {
    local NODE2_IP="${1:?Usage: $0 server1 <node2_internal_ip>}"
    activate_env

    echo "Starting DHT on port $DHT_PORT ..."
    python3 -c "
import hivemind, time, sys
dht = hivemind.DHT(host_maddrs=['/ip4/0.0.0.0/tcp/$DHT_PORT'], start=True)
addr = str(dht.get_visible_maddrs()[0])
print(f'DHT_PEER={addr}', flush=True)
open('/tmp/dht_peer.txt','w').write(addr)
while True: time.sleep(1)
" &
    DHT_PID=$!
    sleep 10
    PEER=$(cat /tmp/dht_peer.txt)
    echo "DHT running: $PEER"

    echo "Starting server blocks 0:30 ..."
    exec python3 -u -m bloombee.cli.run_server "$MODEL" \
        --initial_peers "$PEER" \
        --block_indices 0:30 \
        --batch_size 32 --max_batch_size 32 \
        --skip_reachability_check \
        --torch_dtype float16
}

# ---- server2: serve blocks 30:60 ----
cmd_server2() {
    local DHT_PEER="${1:?Usage: $0 server2 <dht_peer_addr>}"
    activate_env

    echo "Starting server blocks 30:60 ..."
    exec python3 -u -m bloombee.cli.run_server "$MODEL" \
        --initial_peers "$DHT_PEER" \
        --block_indices 30:60 \
        --batch_size 32 --max_batch_size 32 \
        --skip_reachability_check \
        --torch_dtype float16
}

# ---- client: run inference test ----
cmd_client() {
    local DHT_PEER="${1:?Usage: $0 client <dht_peer_addr>}"
    activate_env

    echo "Running Gemma4-31B inference ..."
    python3 -u -c "
import torch
from transformers import AutoTokenizer
from bloombee.models.gemma4.config import DistributedGemma4Config
from bloombee.models.gemma4.model import DistributedGemma4ForCausalLM

MODEL = '$MODEL'
PEERS = ['$DHT_PEER']

tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = DistributedGemma4Config.from_pretrained(MODEL)
if isinstance(config, tuple): config = config[0]
config.initial_peers = PEERS
model = DistributedGemma4ForCausalLM.from_pretrained(MODEL, config=config)
model.eval()

inputs = tokenizer('Hi, what is 2+2?', return_tensors='pt')
with torch.inference_mode():
    out = model.generate(inputs['input_ids'], max_new_tokens=20, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
"
}

case "${1:-}" in
    server1) shift; cmd_server1 "$@" ;;
    server2) shift; cmd_server2 "$@" ;;
    client)  shift; cmd_client "$@" ;;
    *)       echo "Usage: $0 {server1|server2|client} [args...]"; exit 1 ;;
esac
