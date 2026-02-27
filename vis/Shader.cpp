#include "Shader.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <glm/gtc/type_ptr.hpp>

static std::string loadShaderFromFile(const std::string& filename) {
    std::ifstream file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static void checkShaderCompilation(GLuint shader, std::string type)
{
    GLint success;
    GLchar infoLog[1024];

    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {
        glGetShaderInfoLog(shader, 1024, NULL, infoLog);
        throw std::runtime_error("Shader (" + type + ") compilation failed: " + infoLog);
    }

}

static void checkProgramLinking(GLuint shader)
{
    GLint success;
    GLchar infoLog[1024];

    glGetProgramiv(shader, GL_LINK_STATUS, &success);

    if (!success) {
        glGetProgramInfoLog(shader, 1024, NULL, infoLog);
        throw std::runtime_error(std::string("Shader program linking failed: ") + infoLog);
    }
}

Shader::Shader() :
    _init(false),
    _shaderProgramID(0),
    _vShaderCode(""),
    _fShaderCode("")
{
}

Shader::~Shader()
{
}

void Shader::init(const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath)
{
    auto loadCompileAttachShader = [&shaderProgramID = this->_shaderProgramID](const std::string& name, const int type, const std::string& path, std::string& code) -> GLuint {
        code = loadShaderFromFile(path);
        const char* shaderCode = code.c_str();
        GLuint id = glCreateShader(type);
        glShaderSource(id, 1, &shaderCode, NULL);
        glCompileShader(id);
        checkShaderCompilation(id, name);
        glAttachShader(shaderProgramID, id);

        return id;
    };

    _shaderProgramID = glCreateProgram();

    GLuint vertexID = loadCompileAttachShader("VERTEX", GL_VERTEX_SHADER, vertexPath, _vShaderCode);
    GLuint fragmentID = loadCompileAttachShader("FRAGMENT", GL_FRAGMENT_SHADER, fragmentPath, _fShaderCode);
    GLuint geometryID = 0;

    if(!geometryPath.empty())
        geometryID = loadCompileAttachShader("GEOMETRY", GL_GEOMETRY_SHADER, geometryPath, _gShaderCode);

    glLinkProgram(_shaderProgramID);
    checkProgramLinking(_shaderProgramID);

    // delete the shaders 
    glDeleteShader(vertexID);
    glDeleteShader(fragmentID);
    
    if (!geometryPath.empty())
        glDeleteShader(geometryID);

    _init = true;
}

void Shader::use() const
{
    if (!_init)
        return;

    glUseProgram(_shaderProgramID);
}

void Shader::setMat4(const std::string& name, const glm::mat4& mat) const
{
    glUniformMatrix4fv(glGetUniformLocation(_shaderProgramID, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}

void Shader::setFloat(const std::string& name, const float& val) const
{
    glUniform1f(glGetUniformLocation(_shaderProgramID, name.c_str()), val);
}