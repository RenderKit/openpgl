// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"

#include <sstream>

namespace rkguide
{

struct DirectionalSampleData
{
    Vector3 position;
    Vector3 direction;
    float weight;
    float pdf;
    float distance;

    const std::string toString() const
    {
        std::stringstream ss;
        oss << "DirectionalSampleData[" << endl
            << "weight = " << weight << endl
            << "position = [" << position[0] << "\t" << position[1] << "\t" << position[2] << "]" << endl
            << "direction = [" << direction[0] << "\t" << direction[1] << "\t" << direction[2] << "]" << endl
            << "distance = " << distance << endl
            << "pdf = " << pdf << endl
            << "]";
        return oss.str();
    }

    bool operator==( const DirectionalSampleData &comp ) const
    {
        return (position    == comp.position    &&
                direction   == comp.direction    &&
                weight      == comp.weight      &&
                pdf         == comp.pdf         &&
                distance    == comp.distance);
    }

    bool operator<( const DirectionalSampleData &comp ) const
    {
        return weight < comp.weight ||
                ( weight        == comp.weight          &&  ( pdf       < comp.pdf              ||
                ( pdf           == comp.pdf             &&  ( distance < comp.distance          ||
                ( distance      == comp.distance        &&  ( position[0] < comp.position[0]    ||
                (position[0]    == comp.position[0]     &&  ( position[1] < comp.position[1]    ||
                (position[1]    == comp.position[1]     &&  ( position[2] < comp.position[2]    ||
                (position[2]    == comp.position[2]     &&  ( direction[0] < comp.direction[0]  ||
                (direction[0]    == comp.direction[0]   &&  ( direction[1] < comp.direction[1]  ||
                (direction[1]    == comp.direction[1]   &&  ( direction[2] < comp.direction[2]  ))))))))))))))));
    }

};

}