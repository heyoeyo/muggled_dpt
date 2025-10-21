

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

function make_mesh_vertex_shader(is_metric_depth = false) {

	/* Helper used to make the vertex shader, with adjustments for handling metric vs. inv. depth mappings */

	// Figure out which depth calculation to use
	const depthnorm_expr = _make_depth_prediction_0to1_str();
	const metricdepth_expr = "a_param + b_param * depth_norm";
	const invreldepth_expr = "1.0 / (a_param + b_param * depth_norm)";
	const depth_value_expr = is_metric_depth ? metricdepth_expr : invreldepth_expr;

	return `#version 300 es

	// Texture which holds depth data for setting vertex z coords.
	uniform sampler2D u_depth_texture;

	// Uniforms to control scaling/orientation
	uniform mat4 u_world_to_screen_matrix;
	uniform vec4 u_camparams;
	uniform vec3 u_scaling_factors;

	// Set up input/output vars
	in vec2 vert_xy;
	out vec2 uv_01;
	out float mask_value;

	// For clarity (unpack provided controls)
	#define a_param             u_camparams.x
	#define b_param             u_camparams.y
	#define tan_fov             u_camparams.z
	#define edge_threshold      u_camparams.w
	#define x_scale             u_scaling_factors.x
	#define y_scale             u_scaling_factors.y
	#define point_size          u_scaling_factors.z

	void main() {

	// Create UVs that go from 0 to 1, with (0,0) in bot-left, (1, 1) in top right
	// -> Original vert_xy goes from -1 to +1 in both x & y
	uv_01 = (vert_xy + 1.0) * 0.5;

	// Sample depth data to form full vertex xyz coordinates & apply scaling
	// -> Assumes depth data is stored in uint24 precision, held in RGB channels, each 8 bits!
	vec4 depth_rgba = texture(u_depth_texture, uv_01);
	float depth_norm = ${depthnorm_expr};
	float depth_value = ${depth_value_expr};

	// Set masking flag (for use in fragment shader), uses mask data held in depth alpha channel!
	mask_value = depth_rgba.a < edge_threshold ? -1000000.0 : 1.0;

	// Compute world xyz according to depth prediction
	float x_world = depth_value * vert_xy.x * x_scale * tan_fov;
	float y_world = depth_value * vert_xy.y * y_scale * tan_fov;
	float z_world = -depth_value;

	// Form final vertex xyz world coords, assuming far objects are in negative-z regions
	vec4 xyz_world = vec4(x_world, y_world, z_world, 1.0);

	// Set vertex position
	gl_Position = u_world_to_screen_matrix * xyz_world;
	gl_PointSize = point_size;
	}
`
}

// ....................................................................................................................

function make_mesh_fragment_shader(use_depth_texture=false) {

	/* Helper used to make the fragment shader, with adjustments for rendering RGB vs. depth (grayscale) */

	// Set up code for rendering either image (RGB) or depth (grayscale) as texture
	const depthpred_expr = _make_depth_prediction_0to1_str();
	const color_from_image = `out_color = texture(u_image_texture, uv_01);`;
	const color_from_depth = [
		`vec4 depth_rgba = texture(u_depth_texture, uv_01);`,
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
	in float mask_value;

	// Shader output (required!)
	out vec4 out_color;

	void main() {
	if (mask_value < 0.0) discard;
	${frag_color_expr};
	}
`
}

// ....................................................................................................................

function _make_depth_prediction_0to1_str(depth_texture_name = "depth_rgba") {
	/*
	Helper which maps 24bit RGB depth into a single 0-to-1 value for shaders
	Assume 24-bit RGB depth is stored as with R (red) as upper-most 8-bits, then G, B:
	  24bit RGB = RRRRRRRR GGGGGGGG BBBBBBBB
	
	Assume shader RGB reads texture as 0-to-1 values
	To convert RGB depth to single float value, we need:
	  depth_24bit = r_0to1 * (binary: 11111111 00000000 00000000)
	              + g_0to1 * (binary: 00000000 11111111 00000000)
				  + b_0to1 * (binary: 00000000 00000000 11111111)
				  = (r_0to1 * 16711680) + (g_0to1 * 65280) + (b_0to1 * 256)
	Then to normalize back to a 0to1 value, we divide by 2^24 (16777216)

	Finally, can save a bit of computation by factoring out the blue multiplier:
	  depth_0to1 = (65280 * r + 255 * g + b) / 65536;
	*/
	return `(65280.0 * depth_rgba.r + 255.0 * depth_rgba.g + depth_rgba.b) / 65536.0`;
}

// ....................................................................................................................

function run_vertex_shader_cpu(vertex_xy_list, shader_uniforms, depth_image_data, is_metric_depth=false) {

	/*
	This function is meant to implement the vertex shader used to displace vertices based on a depth map.
	It is used to generate vertex xyz data for saving as a 3d model.
	Returns:
		vertex_uv_list, vertex_xyz_list
	*/

	// For clarity. Unpack uniforms matching shader code
	const [a_param, b_param, tan_fov, edge_threshold] = shader_uniforms.camera_params;
	const [x_scale, y_scale, point_size] = shader_uniforms.scaling_factors;
	const edge_threshold_uint8 = edge_threshold * 255.0;

	// Choose which depth calculation to use
	const _metric_depth = d => (a_param + b_param * d);
	const _invrel_depth = d => 1.0 / (a_param + b_param * d);
	const calc_depth = is_metric_depth ? _metric_depth : _invrel_depth;

	const vertex_is_valid = [];
	const vertex_uv_list = [];
	const vertex_xyz_list = [];
	for(const [vx, vy] of vertex_xy_list) {

		// Create UVs
		const uv_val = [vx, vy].map(v => (v + 1.0) * 0.5);
		vertex_uv_list.push(uv_val);

		// Convert 24-bit depth image (+ 8bit mask) to a 0-to-1 depth value
		// (unlike shader code, RGBA values are uint8!)
		const [d_red, d_green, d_blue, d_alpha] = _uv_sample_texture(uv_val, depth_image_data);
		const depth_norm = ((d_red << 16) + (d_green << 8) + d_blue) / (2**24);
		const depth_value = calc_depth(depth_norm);

		// Check edge thresholding. Unlike shader code, we don't use varyings, just a flag for filtering later
		vertex_is_valid.push(d_alpha >= edge_threshold_uint8);

		// Compute xyz world coordinates for each vertex
		const x_world = depth_value * vx * x_scale * tan_fov;
		const y_world = depth_value * vy * y_scale * tan_fov;
		const z_world = -depth_value;
		vertex_xyz_list.push([x_world, y_world, z_world]);
	}

	return [vertex_is_valid, vertex_xyz_list, vertex_uv_list];
}

// ....................................................................................................................

function _uv_sample_texture(uv_value, image_data, num_channels=4) {

	/* Helper used to mimic texture sampling used by shaders */

	const max_x = image_data.width - 1;
	const max_y = image_data.height - 1;

	const [u_val, v_val] = uv_value.map(val => Math.max(0, Math.min(1, val)));
	const x_raw = u_val * max_x;
	const y_raw = v_val * max_y;

	const [x1, y1, x2, y2, tx, ty] = _get_bilinear_xys(image_data, x_raw, y_raw);
	const tl_val = _sample_one_pixel(image_data, x1, y1, num_channels);
	const tr_val = _sample_one_pixel(image_data, x2, y1, num_channels);
	const bl_val = _sample_one_pixel(image_data, x1, y2, num_channels);
	const br_val = _sample_one_pixel(image_data, x2, y2, num_channels);

	let l_val, r_val;
	const avg_val = Array(num_channels);
	for(let i = 0; i < num_channels; i++) {
		l_val = (1.0 - ty)*tl_val[i] + ty*bl_val[i];
		r_val = (1.0 - ty)*tr_val[i] + ty*br_val[i];
		avg_val[i] = (1.0 - tx)*l_val + tx*r_val;
	}

	return avg_val;
}

// ....................................................................................................................

function _get_bilinear_xys(image_data, raw_x_index, raw_y_index) {
	/* Helper used to get 4 nearest-neighbor pixels for bilinear sampling */
	const x1 = Math.floor(raw_x_index);
	const x2 = Math.min(x1 + 1, image_data.width - 1);
	const y1 = Math.floor(raw_y_index);
	const y2 = Math.min(y1 + 1, image_data.height - 1);

	const tx = raw_x_index - x1;
	const ty = raw_y_index - y1;
	return [x1, y1, x2, y2, tx, ty];
}

// ....................................................................................................................

function _sample_one_pixel(image_data, x_idx, y_idx, num_channels=4) {
	/* Helper that reads a single pixel value (e.g. RGBA) at a given xy coord. */
	const pixel_idx = (y_idx * image_data.width + x_idx) * num_channels;
	const output = Array(num_channels);
	for(let i=0; i<num_channels; i++) {
		output[i] = image_data.data[pixel_idx + i];
	}
	return output;
}

// ....................................................................................................................

function make_frustrum_vertex_shader(is_metric_depth=false) {
	/* Shader used to map frustrum into mesh coordinate system */

	return `#version 300 es

	// Uniforms to control scaling/orientation
	uniform mat4 u_world_to_screen_matrix;
	uniform vec4 u_camparams;
	uniform vec3 u_scaling_factors;

	in vec3 vert_xyz;

	// For clarity (unpack provided controls)
	#define min_depth           u_camparams.x
	#define max_depth           u_camparams.y
	#define tan_fov             u_camparams.z
	#define x_scale             u_scaling_factors.x
	#define y_scale             u_scaling_factors.y

	void main() {

	// Map depth values to 0, min or max by value
	float depth_value;
	if (vert_xyz.z > 1.5) {
	  depth_value = max_depth;
	} else if(vert_xyz.z > 0.5) {
	  depth_value = min_depth;
	} else {
	  depth_value = 0.0;
	}

	// Compute world xyz according to depth prediction
	float x_world = depth_value * vert_xyz.x * x_scale * tan_fov;
	float y_world = depth_value * vert_xyz.y * y_scale * tan_fov;
	float z_world = -depth_value;

	vec4 xyz_world = vec4(x_world, y_world, z_world, 1.0);
	gl_Position = u_world_to_screen_matrix * xyz_world;
	}
`
}


function make_frustrum_fragment_shader() {
	/* Shader used to render a camera frustrum indicator */

	return `#version 300 es

	// Set precision
	precision mediump float;

	// Shader output (required!)
	out vec4 out_color;

	void main() {
	  out_color = vec4(0.95, 0.65, 0.0, 1.0);
	}
`
}
