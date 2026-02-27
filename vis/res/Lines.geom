#version 330 core

// Input
layout (lines) in;
layout (triangle_strip, max_vertices = 4) out;
in vec4 gColor[];
in float lineWidthFactor[];

// Output
out vec4  vColor;

void main()
{
    vec3 dir = normalize(gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz);
    vec3 normal = vec3(-dir.y, dir.x, 0.0);

    vec4 offset = vec4(normal * 0.001 * lineWidthFactor[0], 0.0);

    gl_Position = gl_in[0].gl_Position + offset;
    vColor = gColor[0];
    EmitVertex();

    gl_Position = gl_in[0].gl_Position - offset;
    vColor = gColor[0];
    EmitVertex();

    gl_Position = gl_in[1].gl_Position + offset;
    vColor = gColor[1];
    EmitVertex();

    gl_Position = gl_in[1].gl_Position - offset;
    vColor = gColor[1];
    EmitVertex();

    EndPrimitive();
}

