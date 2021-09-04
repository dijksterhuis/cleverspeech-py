#!/usr/bin/env bash

BASE_IMAGE=dijksterhuis/cleverspeech:latest
docker pull ${BASE_IMAGE}
docker tag ${BASE_IMAGE} dijksterhuis/cleverspeech:icaasp
# ./build.sh
#docker push dijksterhuis/cleverspeech:icaasp

GPU_DEVICE=${1}
EXP_SET=${2}

VOLUME_MOUNTS="$(pwd)/adv/:/home/cleverspeech/cleverSpeech/adv:rw"
FLAGS="-it --rm --gpus device=${GPU_DEVICE}"
USER_FLAGS="-e LOCAL_UID=$(id -u ${USER}) -e LOCAL_GID=$(id -g ${USER})"
DOCKER_CMD="docker run ${FLAGS} ${USER_FLAGS} -v ${VOLUME_MOUNTS}"

BTC=25
EXAMPLES=100
STEPS=20000
DECODE_STEP=50

for RANDOM in 4567 3 3248 62977 99999
do
  NAME=icaasp-${GPU_DEVICE}-${EXP_SET}-${RANDOM}
  ${DOCKER_CMD} --name "${NAME}" ${BASE_IMAGE} \
    python3 ./cleverspeech/scripts/icaasp.py \
      --set "${EXP_SET}" \
      --max_examples ${EXAMPLES} \
      --batch_size ${BTC} \
      --nsteps ${STEPS} \
      --decode_step ${DECODE_STEP} \
      --random_seed ${RANDOM} \
      --outdir ./adv/icaasp/${RANDOM} \
      --delta_randomiser 0.01 \
      --rescale 0.9 \
      --learning_rate 10.0 2>&1 | tee -a "${NAME}.log"
done
