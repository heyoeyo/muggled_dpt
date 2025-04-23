
/*
WebGL helper functions. Mostly used to setup shader code (and associated control values).
This is a bunch of messy code, heavily based off the incredible tutorial:
https://webgl2fundamentals.org/
*/

class GLData {

	/* Class used to hold basic data for webgl + uniform location data */

	constructor(webgl_context, shader_defs) {

		this.gl = webgl_context;
		this._shader_defs = shader_defs;
		this.program = null;

		// Storage for locations of uniforms
		this.uniloc = {
			scaling_factors: null,
			camera_params: null,
			matrix: null,
			mask_threshold: null,
		}
	}

	// ................................................................................................................

	create_new_program = (is_metric_depth=false, use_depth_texture=false) => {

		/* Function used to create a new webgl program (i.e. compile shaders) */

		// Set up shaders for current config and verify varnames exist in source code
		const vert_shader = make_vertex_shader(is_metric_depth);
		const frag_shader = make_fragment_shader(use_depth_texture);
		verify_shader_varnames(this._shader_defs, vert_shader, frag_shader);

		// Compile shaders
		const gl = this.gl;
		const program = _create_webgl_program(gl, vert_shader, frag_shader);
		this.program = program;

		// Enable use of compiled shaders
		gl.useProgram(program);
		gl.enable(WebGLRenderingContext.DEPTH_TEST);
		gl.enable(gl.CULL_FACE);

		// Set up access for loading uniform values into shader
		this.uniloc.scaling_factors = gl.getUniformLocation(program, this._shader_defs.uniforms.scaling_factors);
		this.uniloc.camera_params = gl.getUniformLocation(program, this._shader_defs.uniforms.camera_params);
		this.uniloc.matrix = gl.getUniformLocation(program, this._shader_defs.uniforms.matrix);
		this.uniloc.mask_threshold = gl.getUniformLocation(program, this._shader_defs.uniforms.mask_threshold);

		return program;
	}

	// ................................................................................................................

	read_current_uniform_data = () => {
		/* Helper used to read the current uniform values used by the webgl program */

		const scaling_factors = this.gl.getUniform(this.program, this.uniloc.scaling_factors);
		const camera_params = this.gl.getUniform(this.program, this.uniloc.camera_params);
		const matrix = this.gl.getUniform(this.program, this.uniloc.matrix);
		const mask_threshold = this.gl.getUniform(this.program, this.uniloc.mask_threshold);

		return {scaling_factors, camera_params, matrix, mask_threshold};
	}
}


// ....................................................................................................................

function _create_webgl_program(gl_ref, vertex_shader_str, fragment_shader_str) {
	
	/* Helper used to make a webgl 'program' (made of a vertex & fragment shader) */

	// Compile shaders
	const v_shader = _create_webgl_shader(gl_ref, vertex_shader_str, gl_ref.VERTEX_SHADER);
	const f_shader = _create_webgl_shader(gl_ref, fragment_shader_str, gl_ref.FRAGMENT_SHADER);

	// Create program with shaders
	const program = gl_ref.createProgram();
	gl_ref.attachShader(program, v_shader);
	gl_ref.attachShader(program, f_shader);
	gl_ref.linkProgram(program);
	
	// Check that program was made successfully
	const success = gl_ref.getProgramParameter(program, gl_ref.LINK_STATUS);
	if (!success) {
		console.error("Error creating webgl_ref program:", gl_ref.getProgramInfoLog(program));
		gl_ref.deleteProgram(program);
		return null;
	}
	
	return program;
}

// ....................................................................................................................

function _create_webgl_shader(gl_ref, shader_source_str, shader_type) {

	/* Helper used to compile a single shader given a string as input and vertex/fragment type */

	const shader = gl_ref.createShader(shader_type);
	gl_ref.shaderSource(shader, shader_source_str);
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

// ....................................................................................................................

function verify_shader_varnames(shader_defs, vertex_shader_str, fragment_shader_str) {

	/*
	Sanity check function. Used to make sure shader names match with shader source code
	Assumes shader_defs has a structure like:
	  {
	    attrs: {
		  varname1: "attr_name",
		  varname2: "other_name",
		},
		uniforms: {
		  ctrl_1: "u_control",
		  matrix: "u_matrix",
		  ...
		},
		textures: {
		  ...
		},
		other_things: {
		  ...
		},
		etc.
	  }

	Idea is to check that "attr_name", "other_name", "u_control", "u_matrix", etc.
	appear somewhere in the provided vertex/fragment shaders.
	*/

	// Create merged variable names for convenience
	const flat_varnames = Object.assign({}, ...Object.values(shader_defs));

	// Sanity check. Make sure shader variable can be found inside shader code!
	let is_valid = true;
	for (const shader_var_name of Object.values(flat_varnames)) {
		const is_in_vert = vertex_shader_str.includes(shader_var_name);
		const is_in_frag = fragment_shader_str.includes(shader_var_name);
		const is_valid_var = is_in_vert || is_in_frag;
		if (!is_valid_var) {
			is_valid = false;
			throw new Error(`Unknown shader variable: ${shader_var_name}`);
		}
	}

	return is_valid;
}