
/*
WebGL helper functions. Mostly used to setup shader code (and associated control values).
This is a bunch of messy code, heavily based off the incredible tutorial:
https://webgl2fundamentals.org/
*/

// For clarity, define variable names from the shader code (these have to match the shaders exactly!)
const SHADERDEFS = {
	IN_VERTEX_XY: "vert_xy",
	SCALING_FACTORS_UNIFORM: "u_scaling_factors",
	CAMERA_MATRIX_UNIFORM: "u_world_to_screen_matrix",
	CAMERA_PARAM_UNIFORM: "u_camparams",
	TEXTURE_SELECT_UNIFORM: "u_texture_select",
	IMAGE_TEXTURE_UNIFORM: "u_image_texture",
	DEPTH_TEXTURE_UNIFORM: "u_depth_texture",
};

const DATATYPES = {
	rgb_format: WebGL2RenderingContext.RGB8,
	depth_format: WebGL2RenderingContext.RGB8
}


// ...................................................................................................................

function create_shader(gl_ref, source, type) {
	const shader = gl_ref.createShader(type);
	gl_ref.shaderSource(shader, source);
	gl_ref.compileShader(shader);
	var success = gl_ref.getShaderParameter(shader, gl_ref.COMPILE_STATUS);
	
	// Check that shader was compiled successfully
	if (!success) {
		console.error("Error creating shaders:", gl_ref.getShaderInfoLog(shader));
		gl_ref.deleteShader(shader);
		return null;
	}
	
	return shader;
}

// ...................................................................................................................

function create_program(gl_ref, vertex_shader_str, fragment_shader_str) {
	
	// Compile shaders
	const v_shader = create_shader(gl_ref, vertex_shader_str, gl_ref.VERTEX_SHADER);
	const f_shader = create_shader(gl_ref, fragment_shader_str, gl_ref.FRAGMENT_SHADER);

	// Create program with shaders
	const program = gl_ref.createProgram();
	gl_ref.attachShader(program, v_shader);
	gl_ref.attachShader(program, f_shader);
	gl_ref.linkProgram(program);
	
	// Check that program was made successfully
	const success = gl_ref.getProgramParameter(program, gl_ref.LINK_STATUS);
	if (!success) {
		console.error("Error creating webGL program:", gl_ref.getProgramInfoLog(program));
		gl_ref.deleteProgram(program);
		return null;
	}
	
	return program;
}

// ...................................................................................................................

function init_attribute_data(gl_ref, gl_program, vertex_xys, triangle_index_list) {

	// Make sure we setup for the proper shaders
	gl_ref.useProgram(gl_program);

	// Useful info
	const num_verts = vertex_xys.length;
	const num_dimensions = vertex_xys[0].length;
	const num_tris = triangle_index_list.length;
	const num_elements = num_tris * 3;

	// Get reference to attribute variables defined in shader code
	const position_attr_loc = gl_ref.getAttribLocation(gl_program, SHADERDEFS.IN_VERTEX_XY);

	// Pass data into buffer
	const position_data_buffer = gl_ref.createBuffer();
	gl_ref.bindBuffer(gl_ref.ARRAY_BUFFER, position_data_buffer);
	gl_ref.bufferData(gl_ref.ARRAY_BUFFER, new Float32Array(vertex_xys.flat()), gl_ref.STATIC_DRAW);

	// Create vertex array object (not needed?)
	const vao = gl_ref.createVertexArray();
	gl_ref.bindVertexArray(vao);
	gl_ref.enableVertexAttribArray(position_attr_loc);

	// Specify how attribute data (xy coords) is formatted within the buffer for reading in vertex shader
	const attr_size = num_dimensions;
	const attr_type = gl_ref.FLOAT;
	const normalize = false;
	const buffer_stride = 0;
	const buffer_offset = 0;
	gl_ref.vertexAttribPointer(position_attr_loc, attr_size, attr_type, normalize, buffer_stride, buffer_offset);

	// Bind triangle indexing data
	const vert_index_buffer = gl_ref.createBuffer();
	gl_ref.bindBuffer(gl_ref.ELEMENT_ARRAY_BUFFER, vert_index_buffer);
	gl_ref.bufferData(gl_ref.ELEMENT_ARRAY_BUFFER, new Uint32Array(triangle_index_list.flat()), gl_ref.STATIC_DRAW);

	// Bundle attribute data for re-use
	return {
		"vao": vao,
		"buffer": position_data_buffer,
		"loc": position_attr_loc,
		"num_verts": num_verts,
		"num_dimensions": num_dimensions,
		"num_tris": num_tris,
		"num_elements": num_elements,
	};
}

// ...................................................................................................................

function init_texture_data(gl_ref, gl_program) {

	// Make sure we setup for the proper shaders
	gl_ref.useProgram(gl_program);

	// Set up the rgb image texture
	const gl_image_texture = gl_ref.createTexture();
	const gl_image_unit = 0;
	gl_ref.activeTexture(gl_ref.TEXTURE0 + gl_image_unit);
	gl_ref.bindTexture(gl_ref.TEXTURE_2D, gl_image_texture);
	gl_ref.texParameteri(gl_ref.TEXTURE_2D, gl_ref.TEXTURE_WRAP_S, gl_ref.CLAMP_TO_EDGE);
	gl_ref.texParameteri(gl_ref.TEXTURE_2D, gl_ref.TEXTURE_WRAP_T, gl_ref.CLAMP_TO_EDGE);
	gl_ref.texParameteri(gl_ref.TEXTURE_2D, gl_ref.TEXTURE_MIN_FILTER, gl_ref.LINEAR_MIPMAP_LINEAR);
	gl_ref.texParameteri(gl_ref.TEXTURE_2D, gl_ref.TEXTURE_MAG_FILTER, gl_ref.LINEAR);

	// Set up depth texture
	const gl_depth_texture = gl_ref.createTexture();
	const gl_depth_unit = 1;
	gl_ref.activeTexture(gl_ref.TEXTURE0 + gl_depth_unit);
	gl_ref.bindTexture(gl_ref.TEXTURE_2D, gl_depth_texture);
	gl_ref.texParameteri(gl_ref.TEXTURE_2D, gl_ref.TEXTURE_WRAP_S, gl_ref.CLAMP_TO_EDGE);
	gl_ref.texParameteri(gl_ref.TEXTURE_2D, gl_ref.TEXTURE_WRAP_T, gl_ref.CLAMP_TO_EDGE);
	gl_ref.texParameteri(gl_ref.TEXTURE_2D, gl_ref.TEXTURE_MIN_FILTER, gl_ref.LINEAR_MIPMAP_LINEAR);
	gl_ref.texParameteri(gl_ref.TEXTURE_2D, gl_ref.TEXTURE_MAG_FILTER, gl_ref.LINEAR);

	// Set up uniform naming, so we can access textures separately in shaders
	const image_loc = gl_ref.getUniformLocation(gl_program, SHADERDEFS.IMAGE_TEXTURE_UNIFORM);
	const depth_loc = gl_ref.getUniformLocation(gl_program, SHADERDEFS.DEPTH_TEXTURE_UNIFORM);
	gl_ref.uniform1i(image_loc, gl_image_unit);
	gl_ref.uniform1i(depth_loc, gl_depth_unit);
	
	const init_textures = (image_wh, depth_wh, num_mipmaps=3) => {
		gl_ref.activeTexture(gl_ref.TEXTURE0 + gl_image_unit);
		gl_ref.bindTexture(gl_ref.TEXTURE_2D, gl_image_texture);
		gl_ref.texStorage2D(gl_ref.TEXTURE_2D, num_mipmaps, gl_ref.RGB8, image_wh[0], image_wh[1]);

		gl_ref.activeTexture(gl_ref.TEXTURE0 + gl_depth_unit);
		gl_ref.bindTexture(gl_ref.TEXTURE_2D, gl_depth_texture);
		gl_ref.texStorage2D(gl_ref.TEXTURE_2D, num_mipmaps, gl_ref.RGB8, depth_wh[0], depth_wh[1]);
	}

	// Define update function, used to send new RGB & depth textures to the GPU
	const update_textures = (image_bitmap, depth_bitmap) => {

		gl_ref.activeTexture(gl_ref.TEXTURE0 + gl_image_unit);
		gl_ref.bindTexture(gl_ref.TEXTURE_2D, gl_image_texture);
		gl_ref.texSubImage2D(gl_ref.TEXTURE_2D, 0, 0, 0, image_bitmap.width, image_bitmap.height, gl_ref.RGB, gl_ref.UNSIGNED_BYTE, image_bitmap);
		gl_ref.generateMipmap(gl_ref.TEXTURE_2D);

		gl_ref.activeTexture(gl_ref.TEXTURE0 + gl_depth_unit);
		gl_ref.bindTexture(gl_ref.TEXTURE_2D, gl_depth_texture);
		gl_ref.texSubImage2D(gl_ref.TEXTURE_2D, 0, 0, 0, depth_bitmap.width, depth_bitmap.height, gl_ref.RGB, gl_ref.UNSIGNED_BYTE, depth_bitmap);
		gl_ref.generateMipmap(gl_ref.TEXTURE_2D);
	}

	return {
		"image": {"unit": gl_image_unit, "texture": gl_image_texture, "loc": image_loc},
		"depth": {"unit": gl_depth_unit, "texture": gl_depth_texture, "loc": depth_loc},
		"init_textures": init_textures,
		"update_textures": update_textures,
	};
}

// ...................................................................................................................

function init_uniform_data(gl_ref, gl_program) {

	// Make sure we setup for the proper shaders
	gl_ref.useProgram(gl_program);

	const scaling_loc = gl_ref.getUniformLocation(gl_program, SHADERDEFS.SCALING_FACTORS_UNIFORM);
	const matrix_loc = gl_ref.getUniformLocation(gl_program, SHADERDEFS.CAMERA_MATRIX_UNIFORM);
	const camparam_loc = gl_ref.getUniformLocation(gl_program, SHADERDEFS.CAMERA_PARAM_UNIFORM);
	const uv_select_loc = gl_ref.getUniformLocation(gl_program, SHADERDEFS.TEXTURE_SELECT_UNIFORM);

	return {
		"scaling_loc": scaling_loc,
		"matrix_loc": matrix_loc,
		"camparam_loc": camparam_loc,
		"uv_select_loc": uv_select_loc,
	};
}