#!/bin/bash
## Copyright 2020-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

#### Helper functions ####

function check_symbols
{
  for sym in `nm $1 | grep $2_`
  do
    version=(`echo $sym | sed 's/.*@@\(.*\)$/\1/g' | grep -E -o "[0-9]+"`)
    if [ ${#version[@]} -ne 0 ]; then
      if [ ${#version[@]} -eq 1 ]; then version[1]=0; fi
      if [ ${#version[@]} -eq 2 ]; then version[2]=0; fi
      if [ ${version[0]} -gt $3 ]; then
        echo "Error: problematic $2 symbol " $sym
        exit 1
      fi
      if [ ${version[0]} -lt $3 ]; then continue; fi

      if [ ${version[1]} -gt $4 ]; then
        echo "Error: problematic $2 symbol " $sym
        exit 1
      fi
      if [ ${version[1]} -lt $4 ]; then continue; fi

      if [ ${version[2]} -gt $5 ]; then
        echo "Error: problematic $2 symbol " $sym
        exit 1
      fi
    fi
  done
}

function check_imf
{
for lib in "$@"
do
  if [ -n "`ldd $lib | fgrep libimf.so`" ]; then
    echo "Error: dependency to 'libimf.so' found"
    exit 3
  fi
done
}

#### Set variables for script ####

ROOT_DIR=$PWD

DEP_BUILD_DIR=$ROOT_DIR/build_deps
DEP_INSTALL_DIR=$ROOT_DIR/install_deps

OPENPGL_PKG_BASE=openpgl-${OPENPGL_RELEASE_PACKAGE_VERSION}.x86_64.linux
OPENPGL_BUILD_DIR=$ROOT_DIR/build_release
OPENPGL_INSTALL_DIR=$ROOT_DIR/install_release/$OPENPGL_PKG_BASE

THREADS=`nproc`

#### Cleanup any existing directories ####

rm -rf $DEP_INSTALL_DIR
rm -rf $DEP_BUILD_DIR
rm -rf $OPENPGL_BUILD_DIR
rm -rf $OPENPGL_INSTALL_DIR

#### Build dependencies ####

mkdir $DEP_BUILD_DIR
cd $DEP_BUILD_DIR

# NOTE(jda) - Some Linux OSs need to have lib/ on LD_LIBRARY_PATH at build time
export LD_LIBRARY_PATH=$DEP_INSTALL_DIR/lib:${LD_LIBRARY_PATH}

cmake --version

cmake \
  "$@" \
  -D BUILD_DEPENDENCIES_ONLY=ON \
  -D CMAKE_INSTALL_PREFIX=$DEP_INSTALL_DIR \
  -D CMAKE_INSTALL_LIBDIR=lib \
  ../superbuild

cmake --build .

cd $ROOT_DIR

#### Build Open VKL ####

mkdir -p $OPENPGL_BUILD_DIR
cd $OPENPGL_BUILD_DIR

# Setup environment variables for dependencies
#export rkcommon_DIR=$DEP_INSTALL_DIR
export embree_DIR=$DEP_INSTALL_DIR
#export glfw3_DIR=$DEP_INSTALL_DIR

export OPENPGL_EXTRA_OPENVDB_OPTIONS="-DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON"

# set release settings
cmake -L \
  -D CMAKE_INSTALL_PREFIX=$OPENPGL_INSTALL_DIR \
  -D CMAKE_INSTALL_INCLUDEDIR=include \
  -D CMAKE_INSTALL_LIBDIR=lib \
  -D CMAKE_INSTALL_DOCDIR=doc \
  -D CMAKE_INSTALL_BINDIR=bin \
  -D TBB_ROOT=$DEP_INSTALL_DIR \
  ..

# build
make -j $THREADS install

# verify libs
check_symbols $OPENPGL_INSTALL_DIR/lib/libopenpgl.so GLIBC   2 17 0
check_symbols $OPENPGL_INSTALL_DIR/lib/libopenpgl.so GLIBCXX 3 4 19
check_symbols $OPENPGL_INSTALL_DIR/lib/libopenpgl.so CXXABI  1 3 7

check_imf $OPENPGL_INSTALL_DIR/lib/libopenpgl.so


# copy dependent libs into the install
INSTALL_LIB_DIR=$OPENPGL_INSTALL_DIR/lib

cp -P $DEP_INSTALL_DIR/lib/lib*.so* $INSTALL_LIB_DIR

# tar up the results
cd $OPENPGL_INSTALL_DIR/..
tar -caf $OPENPGL_PKG_BASE.tar.gz $OPENPGL_PKG_BASE
mv *.tar.gz $ROOT_DIR
