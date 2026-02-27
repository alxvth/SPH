#version 330 core

// Input
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aWidth;

// Output
out vec4  gColor;
out float lineWidthFactor;

uniform mat4 view;
uniform mat4 projection;
uniform float opacity;

void main()
{
	gl_Position	 = projection * view * vec4(aPos, 1.0f);
	lineWidthFactor = aWidth; 

	gColor = vec4(aColor, opacity);
}

