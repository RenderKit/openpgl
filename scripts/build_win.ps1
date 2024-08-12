
md build
cd build

cmake --version

cmake -L `-G $args ` -D CMAKE_INSTALL_LIBDIR=lib -D BUILD_PYTHON=OFF ../superbuild

cmake --build . --verbose --config Release --target all

exit $LASTEXITCODE