SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck disable=SC2046
PROJECTPATH=$(dirname $(dirname $SCRIPTPATH))

PYTHONPATH=$PROJECTPATH:$PYTHONPATH \
LRU_CACHE_CAPACITY=1 \
python trainer/train_experiment.py mame hrvs resnet18 1 0.00001 50 --retrain mame_hrvs_resnet18.ckpt