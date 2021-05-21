@echo off
rem Copyright 2021 Intel Corporation
rem SPDX-License-Identifier: Apache-2.0

setlocal

md build
cd build

cmake --version

cmake -L ^
-G "%~1" ^
-T "%~2" ^
-D CMAKE_INSTALL_LIBDIR=lib ^
-D BUILD_PYTHON=OFF ^
%~3 %~4 %~5 %~6 %~7 %~8 %~9 ^
../superbuild

cmake --build . --verbose --config Release --target ALL_BUILD -- /m /nologo

:abort
endlocal
:end

rem propagate any error to calling PowerShell script:
exit
