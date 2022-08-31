#!/bin/bash
## Copyright 2020-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

#### Helper functions ####

umask=`umask`
function onexit {
  umask $umask
}
trap onexit EXIT
umask 002

#### Set variables for script ####

ROOT_DIR=$PWD

DEP_BUILD_DIR=$ROOT_DIR/build_deps
DEP_INSTALL_DIR=$ROOT_DIR/install_deps

OPENPGL_PKG_BASE=openpgl-${OPENPGL_RELEASE_PACKAGE_VERSION}.x86_64.macos
OPENPGL_BUILD_DIR=$ROOT_DIR/build_release
OPENPGL_INSTALL_DIR=$ROOT_DIR/install_release/$OPENPGL_PKG_BASE

THREADS=`sysctl -n hw.logicalcpu`

# to make sure we do not include nor link against wrong TBB
unset CPATH
unset LIBRARY_PATH
unset DYLD_LIBRARY_PATH

#### Cleanup any existing directories ####

rm -rf $DEP_INSTALL_DIR
rm -rf $DEP_BUILD_DIR
rm -rf $OPENPGL_BUILD_DIR
rm -rf $OPENPGL_INSTALL_DIR

#### Build dependencies ####

mkdir $DEP_BUILD_DIR
cd $DEP_BUILD_DIR

cmake --version

cmake \
  "$@" \
  -D BUILD_DEPENDENCIES_ONLY=ON \
  -D CMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -D CMAKE_INSTALL_PREFIX=$DEP_INSTALL_DIR \
  -D TBB_VERSION=2021.1.1 \
  -D TBB_HASH="" \
  -D CMAKE_INSTALL_LIBDIR=lib \
  ../superbuild

cmake --build .

cd $ROOT_DIR

#### Build Open PGL ####

mkdir -p $OPENPGL_BUILD_DIR
cd $OPENPGL_BUILD_DIR

# Setup environment variables for dependencies
#export rkcommon_DIR=$DEP_INSTALL_DIR
export embree_DIR=$DEP_INSTALL_DIR
#export glfw3_DIR=$DEP_INSTALL_DIR

export OPENPGL_EXTRA_OPENVDB_OPTIONS="-DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON"

# set release settings
cmake -L \
  -D CMAKE_PREFIX_PATH="$DEP_INSTALL_DIR\lib\cmake" \
  -D CMAKE_INSTALL_PREFIX=$OPENPGL_INSTALL_DIR \
  -D CMAKE_INSTALL_INCLUDEDIR=include \
  -D CMAKE_INSTALL_LIBDIR=lib \
  -D CMAKE_INSTALL_DOCDIR=doc \
  -D CMAKE_INSTALL_BINDIR=bin \
  -D TBB_ROOT=$DEP_INSTALL_DIR \
  -D TBB_VERSION=2021.1.1 \
  -D TBB_HASH="" \
  ..

# build
make -j $THREADS install

# copy dependent libs into the install
INSTALL_LIB_DIR=$OPENPGL_INSTALL_DIR/lib

cp -P $DEP_INSTALL_DIR/lib/lib*.dylib* $INSTALL_LIB_DIR

# zip up the results
cd $OPENPGL_INSTALL_DIR/..
zip -ry $OPENPGL_PKG_BASE.zip $OPENPGL_PKG_BASE
mv *.zip $ROOT_DIR
