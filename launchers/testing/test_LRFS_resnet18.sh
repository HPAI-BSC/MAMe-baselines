SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck disable=SC2046
PROJECTPATH=$(dirname $(dirname $SCRIPTPATH))

PYTHONPATH=$PROJECTPATH:$PYTHONPATH \
python baselines/trainer/test_baseline_model.py mame lrfs resnet18 128 \
$PROJECTPATH/model_checkpoints/mame_lrfs_resnet18_e43.ckpt