#pragma once

#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Ray.h"

struct Camera
{
    enum EWorldUp
    {
        X_UP = 0,
        Y_UP,
        Z_UP
    };

    Camera() = default;

    void init(glm::vec3 origin, glm::vec3 target, glm::vec3 up, float fov)
    {
        m_initOrigin = origin;
        m_initTarget = target;
        m_initUp = up;
        m_initFov = fov;

        m_fov = fov;

        m_transform = glm::lookAt(origin, target, up);
        m_inv_transform = glm::inverse(m_transform);

        m_origin = origin;

        m_front = glm::normalize(target - origin);

        m_upType = Y_UP;
        if (std::abs(up[0]) > std::abs(up[1]) && std::abs(up[0]) > std::abs(up[2]))
            m_upType = X_UP;
        else if (std::abs(up[2]) > std::abs(up[1]))
            m_upType = Z_UP;
        m_worldUp = glm::vec3(0, 0, 0);
        m_worldUp[m_upType] = up[m_upType] > 0 ? 1.f : -1.f;

        glm::vec4 view = glm::vec4(0, 0, -1, 0);
        view = m_inv_transform * view;
        glm::mat4 rot(1.f);
        switch (m_upType)
        {
            case X_UP:
            {
                // std::cout << "X_UP" << std::endl;
                // rot = glm::rotate(rot, glm::radians(-90.f), glm::vec3(0,1,0));
                break;
            }
            case Y_UP:
            {
                // std::cout << "Y_UP" << std::endl;
                m_pitch = glm::degrees(std::acos(view[1]));
                m_yaw = glm::degrees(std::atan2(view[2], view[0]));
                break;
            }
            case Z_UP:
            {
                // std::cout << "Z_UP" << std::endl;
                m_pitch = glm::degrees(std::acos(view[2]));
                m_yaw = glm::degrees(std::atan2(view[1], view[0]));
                break;
            }
        }
    }

    void updateProjection(float width, float height, float fov, float nearClip = 0.1f, float farClip = 100000.f)
    {
        m_projection = glm::perspective(glm::radians(fov), width / height, nearClip, farClip);
        m_fov = fov;

        m_width = width;
        m_height = height;
        m_nearClip = nearClip;
        m_farClip = farClip;
        m_viewport = glm::vec4(0.f, 0.f, width, height);
    }

    void reset()
    {
        init(m_initOrigin, m_initTarget, m_initUp, m_initFov);
    }

    void update()
    {
        m_transform = glm::lookAt(m_origin, m_origin + m_front, m_worldUp);
        m_inv_transform = glm::inverse(m_transform);
        m_projection = glm::perspective(glm::radians(m_fov), m_width / m_height, m_nearClip, m_farClip);
    }

    Ray getRay(float x, float y) const
    {
        Ray ray;
        ray.origin = m_origin;

        glm::mat4x4 mvp = m_transform * m_projection;
        glm::mat4x4 inv_mvp = glm::inverse(mvp);

        float cx = (x / m_width) * 2.f - 1.f;
        float cy = (y / m_height) * 2.f - 1.f;

        glm::vec4 cdir = glm::inverse(m_projection) * glm::vec4(cx, -cy, 0.0, 1.0);
        glm::vec4 cpos = glm::inverse(m_transform) * glm::vec4(cdir.x, cdir.y, cdir.z, 1.0);
        glm::vec3 dir = glm::normalize(glm::vec3(cpos.x, cpos.y, cpos.z) - m_origin);

        ray.direction = dir;
        ray.tmin = 0.f;
        ray.tmax = 1e6f;

        return ray;
    }

    void getPixel(const glm::vec3 &p, float &x, float &y)
    {
        glm::mat4x4 mvp = m_projection * m_transform;
        glm::vec4 cp = m_projection * (m_transform * glm::vec4(p.x, p.y, p.z, 1.f));
        // std::cout << "cp.x: " << cp.x << "\tcp.y: "<< cp.y << std::endl;
        cp /= cp.w;
        // std::cout << "cp.x: " << cp.x << "\tcp.y: "<< cp.y << std::endl;
        x = ((cp.x + 1.f) * 0.5f) * m_width;
        y = ((-cp.y + 1.f) * 0.5f) * m_height;
    }

    glm::mat4x4 m_transform;
    glm::mat4x4 m_inv_transform;
    glm::mat4x4 m_projection;
    glm::vec4 m_viewport;
    float m_fov;

    float m_width;
    float m_height;
    float m_nearClip;
    float m_farClip;

    glm::vec3 m_origin;
    glm::vec3 m_worldUp;
    glm::vec3 m_front;

    float m_sensitivitySpatial{10.0};

    float m_yaw;
    float m_pitch;

    EWorldUp m_upType;

    glm::vec3 m_initOrigin;
    glm::vec3 m_initUp;
    glm::vec3 m_initTarget;
    float m_initFov;
};