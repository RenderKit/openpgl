#pragma once

#include <glm/vec3.hpp>
#include <iostream>
#include <string>

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;

    float tmin;
    float tmax;
};

inline bool ClosestDistance(const Ray &ray, const glm::vec3 &point, float &dist)
{
    // std::cout << "origin: " << ray.origin.x << "\t"<< ray.origin.y <<"\t"<< ray.origin.z<< "\tp: " << point.x << "\t"<< point.y <<"\t"<< point.z << std::endl;
    glm::vec3 difference = point - ray.origin;
    float d = glm::length(difference);
    // std::cout << "d: " << d << std::endl;
    // std::cout << "origin: " << difference.x << "\t"<< difference.y <<"\t"<< difference.z<< std::endl;
    // std::cout << "direction: " << ray.direction.x << "\t"<< ray.direction.y <<"\t"<< ray.direction.z<< std::endl;
    glm::vec3 toPoint = difference / d;
    float other = glm::dot(toPoint, ray.direction) * d;
    // std::cout << "dot: " << glm::dot(toPoint, ray.direction) << std::endl;
    dist = sqrt(d * d - other * other);
    return true;
}