#!/bin/bash
clang-format -style=file -i -fallback-style=none $(git ls-files -- *.cpp *.h ':!:./third-party' ':!:./tools/openpgl-viewer/third-party')