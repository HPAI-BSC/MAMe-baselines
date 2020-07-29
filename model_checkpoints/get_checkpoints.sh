SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck disable=SC2164
cd $SCRIPTPATH
CKPT_URL=https://storage.hpai.bsc.es/mame-dataset/MAMe_checkpoints.zip
echo "Donwloading..." && wget $CKPT_URL &&
echo "Checksum validation..." && echo "5a543d45aa706fe3cd10d9cd95635cfe $(basename $CKPT_URL)" | md5sum -c &&
echo "Decompressing..." && unzip -o "$(basename $CKPT_URL)" &&
echo "Done!"
