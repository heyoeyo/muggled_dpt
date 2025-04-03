

/*
These are the vertex & fragment shaders used to render the
depth predictioin as a 3d surface. These shaders assume that
a 'plane mesh' is used for rendering!

The vertex shader simply adjusts the z-coordinate of vertices
in the plane mesh based on the value of the depth prediction
at the corresponding vertex/pixel positions.

The fragment shader is used to apply different 'textures'
to the plane mesh surface. For example, applying the RGB
image colors or the depth map itself.
*/

const vertex_shader_str = `#version 300 es

// Texture which holds depth data for setting vertex z coords.
uniform sampler2D u_depth_texture;

// Uniforms to control scaling/orientation
uniform mat4 u_world_to_screen_matrix;
uniform vec4 u_camparams;
uniform vec4 u_scaling_factors;

in vec2 vert_xy;
out vec2 uv_01;

// For clarity (unpack provided controls)
#define a_param             u_camparams.x
#define b_param             u_camparams.y
#define half_fov_rad        u_camparams.z
#define depth_offset        u_camparams.w
#define x_scale             u_scaling_factors.x
#define y_scale             u_scaling_factors.y
#define near_plane_dist     u_scaling_factors.z
#define point_size          u_scaling_factors.w

void main() {

// Create UVs that go from 0 to 1, with (0,0) in bot-left, (1, 1) in top right
// -> Original vert_xy goes from -1 to +1 in both x & y
uv_01 = (vert_xy + 1.0) * 0.5;

// Sample depth data to form full vertex xyz coordinates & apply scaling
// -> Assumes depth data is stored in uint24 precision, held in RGB channels, each 8 bits!
vec3 depth_rgb = texture(u_depth_texture, uv_01).rgb;
float inv_depth = (65535.0 * depth_rgb.r + 256.0 * depth_rgb.g + depth_rgb.b) / 65536.0;
float depth_z = 1.0 / (a_param + b_param * inv_depth);

// Compute aspect-ratio corrected vertex x/y coords
float x_arcor = vert_xy.x * x_scale;
float y_arcor = vert_xy.y * y_scale;

// Assume vertex positions (in -1 to +1 range) represent 'angular offset' from the camera viewing axis
// -> E.g left-most pixel is FOVx/2 degrees away from center, the top-most pixel is FOVy/2 degrees away from center
// -> We use this to convert from vert. position to 'world' xy coordinates
float adj_depth = depth_z - near_plane_dist;
float x_world = x_arcor + adj_depth * tan(x_arcor * half_fov_rad);
float y_world = y_arcor + adj_depth * tan(y_arcor * half_fov_rad);

// Form final vertex xyz world coords, assuming far objects are in negative-z regions
vec4 xyz_world = vec4(x_world, y_world, depth_offset - depth_z, 1.0);

// Set vertex position
gl_Position = u_world_to_screen_matrix * xyz_world;
gl_PointSize = point_size;
}
`;


const fragment_shader_str = `#version 300 es

// Set precision
precision mediump float;

// Uniform to control which texture is rendered
uniform int u_texture_select;

// Textures for rendering
uniform sampler2D u_image_texture;
uniform sampler2D u_depth_texture;

// Passed in from the vertex shader.
in vec2 uv_01;  

// Shader output (required!)
out vec4 out_color;

void main() {

// Select which texture to use for rendering
bool use_image = (u_texture_select < 1);

out_color = vec4(1,0,1,1);
if (use_image) {
    out_color = texture(u_image_texture, uv_01);
} else {
    // Assume depth is stored in uint24, held in RGB channels each 8 bits!
    vec3 depth_rgb = texture(u_depth_texture, uv_01).rgb;
    float depth_value = (65535.0 * depth_rgb.r + 256.0 * depth_rgb.g + depth_rgb.b) / 65536.0;
    out_color.rgb = vec3(depth_value);
}
}
`;
