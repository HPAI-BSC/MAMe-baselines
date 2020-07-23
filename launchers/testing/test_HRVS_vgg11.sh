SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck disable=SC2046
PROJECTPATH=$(dirname $(dirname $SCRIPTPATH))

PYTHONPATH=$PROJECTPATH:$PYTHONPATH \
LRU_CACHE_CAPACITY=1 \
python baselines/trainer/test_baseline_model.py mame hrvs vgg11 1 \
$PROJECTPATH/model_checkpoints/mame_hrvs_vgg11_e41.ckpt