#!/bin/bash -x
## Copyright 2019-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set -e
KW_SERVER_PATH=$KW_PATH/server
KW_CLIENT_PATH=$KW_PATH/client
export KLOCWORK_LTOKEN=/tmp/ltoken
echo "$KW_SERVER_IP;$KW_SERVER_PORT;$KW_USER;$KW_LTOKEN" > $KLOCWORK_LTOKEN
mkdir -p $CI_PROJECT_DIR/klocwork
log_file=$CI_PROJECT_DIR/klocwork/build.log

mkdir build
cd build

# NOTE(jda) - Some Linux OSs need to have TBB on LD_LIBRARY_PATH at build time
export LD_LIBRARY_PATH=`pwd`/install/lib:${LD_LIBRARY_PATH}

cmake --version

cmake \
  -DBUILD_JOBS=`nproc` \
  -DBUILD_DEPENDENCIES_ONLY=ON \
  -DBUILD_PYTHON=OFF \
  "$@" ../superbuild

cmake --build .

mkdir openpgl_build
cd openpgl_build

DEP_INSTALL_DIR=`pwd`/../install

export rkcommon_DIR=$DEP_INSTALL_DIR
export embree_DIR=$DEP_INSTALL_DIR

cmake \
 -DTBB_ROOT=$DEP_INSTALL_DIR \
 -DBUILD_PYTHON=OFF \
 -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  ../..

# build
$KW_CLIENT_PATH/bin/kwinject make -j `nproc` | tee -a $log_file
$KW_SERVER_PATH/bin/kwbuildproject -v --add-compiler-options "--print-errors --strict --verbose" --classic --url http://$KW_SERVER_IP:$KW_SERVER_PORT/$KW_PROJECT_NAME --tables-directory $CI_PROJECT_DIR/kw_tables kwinject.out | tee -a $log_file
$KW_SERVER_PATH/bin/kwadmin --url http://$KW_SERVER_IP:$KW_SERVER_PORT/ load --force --name build-$CI_JOB_ID $KW_PROJECT_NAME $CI_PROJECT_DIR/kw_tables | tee -a $log_file

cat $log_file
# Store kw build name for check status later
echo "build-$CI_JOB_ID" > $CI_PROJECT_DIR/klocwork/build_name

