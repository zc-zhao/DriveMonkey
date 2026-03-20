
CHECKPOINT=${1}
DATASET=${2}
CHECKPOINT="$(pwd)/${CHECKPOINT}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
echo "CHECKPOINT: ${CHECKPOINT}"

MASTER_PORT=${MASTER_PORT:-63663}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
export NCCL_IB_DISABLE=1
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

torchrun \
--nnodes=1 \
--node_rank=0 \
--master_addr=127.0.0.1 \
--nproc_per_node=${GPUS} \
--master_port=${MASTER_PORT} \
eval/nuscene/evaluate_nuscene_bev.py --checkpoint ${CHECKPOINT} \
--datasets ${DATASET} ${@:3}
