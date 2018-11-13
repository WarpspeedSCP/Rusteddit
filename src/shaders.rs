pub mod basic_vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable
 

//layout(binding = 0) uniform uniformBufferObject {
//    mat4 model;
//    mat4 view;
//    mat4 proj;
//} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColour;
layout(location = 2) in vec2 inTexCoord;
//layout(location = 3) in vec3 inNormal;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 outColor;

//layout(push_constant) vec3 light_vec;

const vec3 light_vec = vec3(1, 0, 0.5);

void main() {
    //gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    gl_Position = vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;

    //vec3 normal = mat3( ubo.model ) * inNormal;
    outColor = inColour;//max( 0.0, dot( normal, light_vec )) + 0.1;
}"
    }

    struct Dummy;
}

pub mod basic_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 fragTexCoord;
layout (location = 1) in vec3 inColour;

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColour;

void main() {
    outColour = texture(texSampler, fragTexCoord) * vec4(inColour, 1.0) ;
}"
    }
    struct Dummy;
}