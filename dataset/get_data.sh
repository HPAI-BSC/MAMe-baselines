SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck disable=SC2164
cd $SCRIPTPATH
DATA_URL=https://storage.hpai.bsc.es/mame-dataset/MAMe_data.zip
[ -e data ] && echo "There already exists a data folder. To run this code, first remove it." && exit
echo "Donwloading..." && wget $DATA_URL &&
echo "Checksum validation..." && echo "39b11ef14dba807faf284f74ec2576d3 $(basename $DATA_URL)" | md5sum -c &&
echo "Decompressing..." && unzip -o "$(basename $DATA_URL)" &&
echo "Done!"
