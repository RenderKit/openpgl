#!/bin/bash -xe
## Copyright 2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <archive>"
  exit 1
fi

ARCHIVE=$1

ROOT_DIR=$PWD

# setup signing environment
SIGN_FILE=""
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     SIGN_FILE=${SIGN_FILE_LINUX};;
    Darwin*)    SIGN_FILE=${SIGN_FILE_MAC};;
    CYGWIN*)    SIGN_FILE=${SIGN_FILE_WINDOWS};;
    MINGW*)     SIGN_FILE=${SIGN_FILE_WINDOWS};;
    *)          SIGN_FILE=""
esac

echo "got SIGN_FILE: ${SIGN_FILE}"

if [ -z "${SIGN_FILE}" ]; then
  echo "could not detect SIGN_FILE"
  exit 1
fi

# unpack archive in tmp directory
rm -rf tmp-sign
mkdir tmp-sign

cp ${ARCHIVE} tmp-sign/

cd tmp-sign

if [[ ${ARCHIVE} == *.zip ]]; then
  unzip ${ARCHIVE}
  ARCHIVE_DIR=`echo ${ARCHIVE} | sed s/.zip//g`
elif [[ ${ARCHIVE} == *.tar.gz ]]; then
  tar -zxvf ${ARCHIVE}
  ARCHIVE_DIR=`echo ${ARCHIVE} | sed s/.tar.gz//g`
else
  echo "unhandled archive format"
  exit 1
fi

echo "archive name: ${ARCHIVE_DIR}"

ls -l ${ARCHIVE}

rm ${ARCHIVE}

# sign files as appropriate
#find -L ./${ARCHIVE_DIR}/bin -type f -exec ${SIGN_FILE} -q -vv {} \;
find -L ./${ARCHIVE_DIR}/lib -type f -name *.dylib -exec ${SIGN_FILE} -q -vv {} \;
find -L ./${ARCHIVE_DIR}/lib -type f -name *.so* -exec ${SIGN_FILE} -q -vv {} \;

# repack archive
if [[ ${ARCHIVE} == *.zip ]]; then
  zip -ry ${ARCHIVE} ${ARCHIVE_DIR}
elif [[ ${ARCHIVE} == *.tar.gz ]]; then
  tar -czf ${ARCHIVE} ${ARCHIVE_DIR}
else
  echo "unhandled archive format"
  exit 1
fi

# replace original archive with signed version
cp ${ARCHIVE} ${ROOT_DIR}/${ARCHIVE}

ls -l ${ARCHIVE} ${ROOT_DIR}/${ARCHIVE}

exit 0