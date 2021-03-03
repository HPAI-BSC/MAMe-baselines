SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck disable=SC2164
cd $SCRIPTPATH
CKPT_URL=https://storage.hpai.bsc.es/mame-dataset/MAMe_pretrained_checkpoints.zip
echo "Donwloading..." && wget $CKPT_URL &&
echo "Checksum validation..." && echo "a1d8046c990d2ea24504d43991f8f0f8 $(basename $CKPT_URL)" | md5sum -c &&
echo "Decompressing..." && unzip -o "$(basename $CKPT_URL)" &&
echo "Done!"
