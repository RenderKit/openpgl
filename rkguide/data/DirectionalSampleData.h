// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"


#include <iostream>
#include <fstream>
#include <sstream>

namespace rkguide
{



struct DirectionalSampleData
{

    enum Flags
    {
        ESplatted = 1<<0, // point does not represent any real scene intersection point
        EInsideVolume = 1<<1 // point does not represent any real scene intersection point
    };

    Point3 position;
    Vector3 direction;
    float weight;
    float pdf;
    float distance;
    uint32_t flags;

    DirectionalSampleData() = default;

    DirectionalSampleData(Point3 _pos, Vector3 _dir, float _weight, float _pdf, float _distance, uint32_t _flags):
        position(_pos), direction(_dir), weight(_weight), pdf(_pdf), distance(_distance), flags(_flags)
    {

    }

    bool isInsideVolume() const
    {
        return (flags & EInsideVolume);
    }

    void setIsInsideVolume()
    {
        flags |= EInsideVolume;
    }

    void unsetIsInsideVolume()
    {
        flags ^= EInsideVolume;
    }

    const std::string toString() const
    {
        std::stringstream ss;
        ss << "DirectionalSampleData[" << std::endl
            << "weight = " << weight << std::endl
            << "position = [" << position[0] << "\t" << position[1] << "\t" << position[2] << "]" << std::endl
            << "direction = [" << direction[0] << "\t" << direction[1] << "\t" << direction[2] << "]" << std::endl
            << "distance = " << distance << std::endl
            << "pdf = " << pdf << std::endl
            << "]" << std::endl;
        return ss.str();
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


DirectionalSampleData *LoadDirectionalSampleData(const std::string fileName, size_t &numData){

    std::ifstream file;
    file.open(fileName, std::ios::binary);

    //size_t numData;
    file.read((char*)&numData, sizeof(size_t));

    DirectionalSampleData *data = new DirectionalSampleData[numData];
    file.read((char*)data, numData*sizeof(DirectionalSampleData));
    file.close();

    return data;
}

void StoreDirectionalSampleData(const std::string fileName, const DirectionalSampleData *data, const size_t &numData){
    std::ofstream file;
    file.open(fileName, std::ios::binary);

    file.write((char*)&numData, sizeof(size_t));
    file.write((char*)&data, numData * sizeof(DirectionalSampleData));
    file.close();
}


}