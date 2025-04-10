

/*
These are the vertex & fragment shaders used to render the
depth prediction as a 3d surface. These shaders assume that
a 'plane mesh' is used for rendering!

The vertex shader adjusts the z-coordinate of vertices
in the plane mesh based on the value of the depth prediction
at the corresponding vertex/pixel positions. It also pushes
the x & y coords. out, based on FOV.

The fragment shader is used to apply different 'textures'
to the plane mesh surface. For example, applying the RGB
image colors or the depth map itself.
*/

// For clarity, define variable names from the shader code (these have to match the shaders exactly!)
const SHADERDEFS = {
	attrs: {
		vertex_xy: "vert_xy",
	},
	uniforms: {
		scaling_factors: "u_scaling_factors",
		camera_params: "u_camparams",
		matrix: "u_world_to_screen_matrix",
	},
	textures: {
		image: "u_image_texture",
		depth: "u_depth_texture",
	},
}

// ....................................................................................................................

function make_vertex_shader(is_metric_depth = false) {

	/* Helper used to make the vertex shader, with adjustments for handling metric vs. inv. depth mappings */

	// Figure out which depth calculation to use
	const depthpred_expr = _make_depth_prediction_str();
	const metricdepth_expr = `(a_param + b_param * ${depthpred_expr})`;
	const invreldepth_expr = `1.0 / ${metricdepth_expr}`;
	const depth_value_expr = is_metric_depth ? metricdepth_expr : invreldepth_expr;

	return `#version 300 es

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
	float depth_value = ${depth_value_expr};

	// Compute aspect-ratio corrected vertex x/y coords
	float x_arcor = vert_xy.x * x_scale;
	float y_arcor = vert_xy.y * y_scale;

	// Assume vertex positions (in -1 to +1 range) represent 'angular offset' from the camera viewing axis
	// -> E.g left-most pixel is FOVx/2 degrees away from center, the top-most pixel is FOVy/2 degrees away from center
	// -> We use this to convert from vert. position to 'world' xy coordinates
	float adj_depth = depth_value - near_plane_dist;
	float x_world = x_arcor + adj_depth * sin(x_arcor * half_fov_rad);
	float y_world = y_arcor + adj_depth * sin(y_arcor * half_fov_rad);

	// Get corrected 'planar depth', assuming prediction gives 'hypotenuse depth'
	float diagonal_dist = sqrt(x_arcor * x_arcor + y_arcor * y_arcor);
	float z_world = cos(diagonal_dist * half_fov_rad) * depth_value;

	// Form final vertex xyz world coords, assuming far objects are in negative-z regions
	vec4 xyz_world = vec4(x_world, y_world, depth_offset - z_world, 1.0);

	// Set vertex position
	gl_Position = u_world_to_screen_matrix * xyz_world;
	gl_PointSize = point_size;
	}
`
}

// ....................................................................................................................

function make_fragment_shader(use_depth_texture=false) {

	/* Helper used to make the fragment shader, with adjustments for rendering RGB vs. depth (grayscale) */

	// Set up code for rendering either image (RGB) or depth (grayscale) as texture
	const depthpred_expr = _make_depth_prediction_str();
	const color_from_image = `out_color = texture(u_image_texture, uv_01);`;
	const color_from_depth = [
		`vec3 depth_rgb = texture(u_depth_texture, uv_01).rgb;`,
		`float depth_value = ${depthpred_expr};`,
		`out_color = vec4(vec3(depth_value), 1.0);`,
	].join("\n");
	const frag_color_expr = use_depth_texture ? color_from_depth : color_from_image;

	return `#version 300 es

	// Set precision
	precision mediump float;

	// Textures for rendering
	uniform sampler2D u_image_texture;
	uniform sampler2D u_depth_texture;

	// Passed in from the vertex shader.
	in vec2 uv_01;

	// Shader output (required!)
	out vec4 out_color;

	void main() {
	${frag_color_expr};
	}
`
}

// ....................................................................................................................

function _make_depth_prediction_str(depth_texture_name = "depth_rgb") {
	/* Helper used to make the expression for converting 24bit RGB depth into 0-to-1 value in shader */
	const red = `${depth_texture_name}.r`;
	const green = `${depth_texture_name}.g`;
	const blue = `${depth_texture_name}.b`;
	return `((65535.0 * ${red} + 256.0 * ${green} + ${blue}) / 65536.0)`;
}

