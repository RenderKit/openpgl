#ifndef opengl_shader_hpp
#define opengl_shader_hpp

#include <string>
#include <vector>
#include <map>

#include <GL/glew.h>

struct Buffer{
	GLuint id;
	GLuint glType;
	GLuint dim;
	GLuint compSize;
	GLuint size;
	int version;	
};

class Shader
{
public:
	Shader();
	void init(const std::string& vertex_code, const std::string& fragment_code);
	void init(const std::string& vertex_code, const std::string& geometry_code, const std::string& fragment_code);
	void bind();
	template<typename T> void setUniform(const std::string& name, T val);
	template<typename T> void setUniform(const std::string& name, T val1, T val2);
	template<typename T> void setUniform(const std::string& name, T val1, T val2, T val3);
	template<typename T> void setUniform(const std::string& name, T val1, T val2, T val3, T val4);

	template<typename T> void uploadAttribute(const std::string& name, T* vals, int size);

private:
	void checkCompileErr();
	void checkLinkingErr();
	void compile();
	void link();
	unsigned int vertex_shader_id, geometry_shader_id, fragment_shader_id, shader_id;
	std::string vertex_code_;
	std::string fragment_code_;
	std::string geometry_code_;
	unsigned int vertex_array_object;

	std::map<std::string, Buffer> buffer_objects;

};

#endif /* opengl_shader_hpp */