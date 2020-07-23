SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck disable=SC2046
PROJECTPATH=$(dirname $(dirname $SCRIPTPATH))

PYTHONPATH=$PROJECTPATH:$PYTHONPATH \
LRU_CACHE_CAPACITY=1 \
python baselines/trainer/test_baseline_model.py mame hrvs resnet18 1 \
$PROJECTPATH/model_checkpoints/mame_hrvs_resnet18_e47.ckpt