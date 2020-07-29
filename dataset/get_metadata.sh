SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck disable=SC2164
cd $SCRIPTPATH
META_URL=https://storage.hpai.bsc.es/mame-dataset/MAMe_metadata.zip
[ ! -e "$(basename $META_URL)" ] && wget $META_URL && echo "0bca980978e3883f753f01ccad955981 $(basename $META_URL)" | md5sum -c && unzip -o "$(basename $META_URL)"
