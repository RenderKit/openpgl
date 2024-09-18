#include "GLShader.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <fstream>
#include <glm/vec3.hpp>
#include <iostream>
#include <sstream>

Shader::Shader() {}

void Shader::init(const std::string &vertex_code, const std::string &fragment_code)
{
    vertex_code_ = vertex_code;
    geometry_code_ = "";
    fragment_code_ = fragment_code;
    compile();
    link();

    glGenVertexArrays(1, &vertex_array_object);
}

void Shader::init(const std::string &vertex_code, const std::string &geometry_code, const std::string &fragment_code)
{
    vertex_code_ = vertex_code;
    geometry_code_ = geometry_code;
    fragment_code_ = fragment_code;
    compile();
    link();

    glGenVertexArrays(1, &vertex_array_object);
}

void Shader::compile()
{
    const char *vcode = vertex_code_.c_str();
    vertex_shader_id = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader_id, 1, &vcode, NULL);
    glCompileShader(vertex_shader_id);

    geometry_shader_id = -1;
    if (geometry_code_ != "")
    {
        const char *gcode = geometry_code_.c_str();
        geometry_shader_id = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometry_shader_id, 1, &gcode, NULL);
        glCompileShader(geometry_shader_id);
    }

    const char *fcode = fragment_code_.c_str();
    fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader_id, 1, &fcode, NULL);
    glCompileShader(fragment_shader_id);
    checkCompileErr();
}

void Shader::link()
{
    shader_id = glCreateProgram();
    glAttachShader(shader_id, vertex_shader_id);
    if (geometry_shader_id != -1)
    {
        glAttachShader(shader_id, geometry_shader_id);
    }
    glAttachShader(shader_id, fragment_shader_id);
    glLinkProgram(shader_id);
    checkLinkingErr();
    glDeleteShader(vertex_shader_id);
    if (geometry_shader_id != -1)
    {
        glDeleteShader(geometry_shader_id);
    }
    glDeleteShader(fragment_shader_id);
}

void Shader::bind()
{
    glUseProgram(shader_id);
    glBindVertexArray(vertex_array_object);
}

template <>
void Shader::setUniform<int>(const std::string &name, int val)
{
    glUniform1i(glGetUniformLocation(shader_id, name.c_str()), val);
}

template <>
void Shader::setUniform<bool>(const std::string &name, bool val)
{
    glUniform1i(glGetUniformLocation(shader_id, name.c_str()), val);
}

template <>
void Shader::setUniform<float>(const std::string &name, float val)
{
    glUniform1f(glGetUniformLocation(shader_id, name.c_str()), val);
}

template <>
void Shader::setUniform<float>(const std::string &name, float val1, float val2)
{
    glUniform2f(glGetUniformLocation(shader_id, name.c_str()), val1, val2);
}

template <>
void Shader::setUniform<float>(const std::string &name, float val1, float val2, float val3)
{
    glUniform3f(glGetUniformLocation(shader_id, name.c_str()), val1, val2, val3);
}

template <>
void Shader::setUniform<float>(const std::string &name, float val1, float val2, float val3, float val4)
{
    glUniform4f(glGetUniformLocation(shader_id, name.c_str()), val1, val2, val3, val4);
}

template <>
void Shader::setUniform<float *>(const std::string &name, float *val)
{
    glUniformMatrix4fv(glGetUniformLocation(shader_id, name.c_str()), 1, GL_FALSE, val);
}

template <>
void Shader::uploadAttribute<glm::vec3>(const std::string &name, glm::vec3 *data, int size)
{
    size_t compSize = sizeof(glm::vec3);
    size_t totalSize = size * (size_t)compSize;
    GLuint bufferID;
    glGenBuffers(1, &bufferID);
    if (name == "indices")
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferID);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, totalSize, data, GL_DYNAMIC_DRAW);
    }
    else
    {
        glBindBuffer(GL_ARRAY_BUFFER, bufferID);
        glBufferData(GL_ARRAY_BUFFER, totalSize, data, GL_DYNAMIC_DRAW);
    }
}

void Shader::checkCompileErr()
{
    int success;
    char infoLog[1024];
    glGetShaderiv(vertex_shader_id, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertex_shader_id, 1024, NULL, infoLog);
        std::cout << "Error compiling Vertex Shader:\n" << infoLog << std::endl;
    }
    glGetShaderiv(fragment_shader_id, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragment_shader_id, 1024, NULL, infoLog);
        std::cout << "Error compiling Fragment Shader:\n" << infoLog << std::endl;
    }
}

void Shader::checkLinkingErr()
{
    int success;
    char infoLog[1024];
    glGetProgramiv(shader_id, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shader_id, 1024, NULL, infoLog);
        std::cout << "Error Linking Shader Program:\n" << infoLog << std::endl;
    }
}