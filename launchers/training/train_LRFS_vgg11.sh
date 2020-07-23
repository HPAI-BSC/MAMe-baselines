SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck disable=SC2046
PROJECTPATH=$(dirname $(dirname $SCRIPTPATH))

PYTHONPATH=$PROJECTPATH:$PYTHONPATH \
python trainer/train_experiment.py mame lrfs vgg11 128 0.00001 50 --retrain mame_lrfs_vgg11.ckpt