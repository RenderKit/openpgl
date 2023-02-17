// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openpgl/cpp/OpenPGL.h>

int main (int argc, char *argv[]) {

    openpgl::cpp::Device device(PGL_DEVICE_TYPE_CPU_4);
    PGLFieldArguments fieldSettings;
    pglFieldArgumentsSetDefaults(fieldSettings, PGL_SPATIAL_STRUCTURE_KDTREE, PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM);
    openpgl::cpp::Field field(&device, fieldSettings);

    return 0;
}