#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>

class Shader
{
public:
	Shader();
	~Shader();

	void init(const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath = {});
	void use() const;
	void setMat4(const std::string& name, const glm::mat4& mat) const;
	void setFloat(const std::string& name, const float& val) const;

	bool isInit() const { return _init; }

private:
	bool			_init;
	GLuint			_shaderProgramID;
	std::string		_vShaderCode;
	std::string		_gShaderCode;
	std::string		_fShaderCode;
};