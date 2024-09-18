// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

enum ShaderType
{
    EShaderVertex = 0,
    EShaderGeometry,
    EShaderFragment,
};

struct ShaderInfo
{
    std::string name{""};
    ShaderType type{EShaderVertex};
    bool valid{false};
    std::ifstream file;
};

std::string base_name(std::string const &path)
{
    return path.substr(path.find_last_of("/\\") + 1);
}

std::string remove_extension(const std::string &baseName)
{
    const int p(baseName.find_last_of('.'));
    return p > 0 && p != std::string::npos ? baseName.substr(0, p) : baseName;
}

std::string get_extension(const std::string &baseName)
{
    const int p(baseName.find_last_of('.'));
    return p > 0 && p != std::string::npos ? baseName.substr(p + 1, baseName.size()) : baseName;
}

ShaderInfo parseShaderInfo(const std::string &shaderFile)
{
    ShaderInfo shaderInfo;
    shaderInfo.file = std::ifstream(shaderFile);
    if (shaderInfo.file.good())
    {
        shaderInfo.valid = true;
        std::string baseName = base_name(shaderFile);
        shaderInfo.name = remove_extension(baseName);

        std::string ext = get_extension(baseName);
        if (ext == "vs")
        {
            shaderInfo.type = EShaderVertex;
        }
        else if (ext == "gs")
        {
            shaderInfo.type = EShaderGeometry;
        }
        else if (ext == "fs")
        {
            shaderInfo.type = EShaderFragment;
        }
        else
        {
            shaderInfo.valid = false;
        }
    }
    return shaderInfo;
}

void printShader(const std::string &shaderFile)
{
    ShaderInfo shaderInfo = parseShaderInfo(shaderFile);
    if (shaderInfo.valid)
    {
        std::cout << "" << "std::string " << shaderInfo.name << "_";
        switch (shaderInfo.type)
        {
            case EShaderVertex:
            {
                std::cout << "vs";
                break;
            }
            case EShaderGeometry:
            {
                std::cout << "gs";
                break;
            }
            case EShaderFragment:
            {
                std::cout << "fs";
                break;
            }
        }
        std::cout << " = { \" \\" << std::endl;
        std::string line;
        while (std::getline(shaderInfo.file, line))
        {
            std::cout << line.c_str() << "\t\\n\\" << std::endl;
        }
        std::cout << "\"};" << std::endl;
    }
}

void printShaders(const std::vector<std::string> &shaderFiles)
{
    std::cout << "#pragma once" << std::endl;
    std::cout << "#include <string>" << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < shaderFiles.size(); i++)
    {
        printShader(shaderFiles[i]);
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 2 && argc != 3 && argc != 4)
    {
        std::cerr << "usage: sc <optinal>.vs <optinal>.gs <optinal>.fs" << std::endl;
        exit(1);
    }

    std::vector<std::string> shaderFiles;
    for (int i = 1; i < argc; i++)
    {
        std::string filename = argv[i];
        shaderFiles.push_back(filename);
    }

    printShaders(shaderFiles);

    return 0;
}
