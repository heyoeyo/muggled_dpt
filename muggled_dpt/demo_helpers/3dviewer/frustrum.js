/*
Contains code used to produce data for lines representing a camera frustrum.
These are rendered into a 3D scene containing a depth prediction, to help
indicate how camera parameters affect the resulting 3D model.
*/


class FrustrumData {

	/* Helper class used to manage vertex data needed for rendering camera frustrum */

	constructor(webgl_context, vertex_attr_name="vert_xyz") {

		// Hold inputs for re-use in data init/rendering
		this._gl = webgl_context;
		this._vert_attr = vertex_attr_name;

		// Storage for holding vertex data
		this._vert_xyz = [];
		this._vert_dimensions = 3;
		this._num_verts = 0;

		this._vert_attr_loc = null;
		this._vao = this._gl.createVertexArray();

		// Buffer for holding vertex xy positions array
		this._vert_buffer_type = WebGL2RenderingContext.ARRAY_BUFFER;
		this._vert_buffer = this._gl.createBuffer();
	}

	// ................................................................................................................

	bind_gl_program = (gl_program) => {

		/* Function used to link frustrum vertex data to a webgl program. Needs to be called if the program changes */

		// For clarity
		const gl = this._gl;
		gl.useProgram(gl_program);

		// Create frustrum vertices if needed
		if (this._num_verts === 0) {
			this._vert_xyz = this._make_frustrum_vertices();
			this._vert_dimensions = this._vert_xyz[0][0].length;
			this._num_verts = this._vert_xyz.flat(Infinity).length;
		}

		// Get reference to vertex attribute defined in shader code
		this._vert_attr_loc = gl.getAttribLocation(gl_program, this._vert_attr);

		// Attach vertex array object to new program
		gl.bindVertexArray(this._vao);
		gl.enableVertexAttribArray(this._vert_attr_loc);

		// Bind vertex data for rendering
		gl.bindBuffer(gl.ARRAY_BUFFER, this._vert_buffer);
      	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(this._vert_xyz.flat(Infinity)), gl.STATIC_DRAW);

		return this;
	}

	// ................................................................................................................

	draw_arrays = (linewidth = 2) => {
		/* Bind buffer data (e.g. frustrum vertices) and draw using current gl program */

		// For convenience
		const gl = this._gl;
		
		// Bind vertex buffer to gl program
		gl.bindBuffer(gl.ARRAY_BUFFER, this._vert_buffer);

		// // Specify how attribute data (xyz coords) is formatted
		const attr_size = 3;
		const attr_type = gl.FLOAT;
		const normalize = false;
		const buffer_stride = 0;
		const buffer_offset = 0;
    	gl.vertexAttribPointer(this._vert_attr_loc, attr_size, attr_type, normalize, buffer_stride, buffer_offset);
    	gl.enableVertexAttribArray(this._vert_attr_loc);

		// Do actual drawing
		gl.lineWidth(linewidth);
		const offset = 0;
		gl.drawArrays(gl.LINES, offset, this._num_verts);

		return this;
	}

	// ................................................................................................................

	_make_frustrum_vertices = () => {
		/* Define camera frustrum vertices. Assume there is a min/max plane, denoted by different z-values */

		// For clarity
		const origin = [0,0,0];
		const min_val = 1;
		const max_val = 2;

		// Set up coords of min-depth 'box'
		const tl_min = [-1, 1, min_val];
		const tr_min = [ 1, 1, min_val];
		const br_min = [ 1,-1, min_val];
		const bl_min = [-1,-1, min_val];

		// Set up coords of max-depth 'box'
		const tl_max = [-1, 1, max_val];
		const tr_max = [ 1, 1, max_val];
		const br_max = [ 1,-1, max_val];
		const bl_max = [-1,-1, max_val];

		// Set up line-drawing verts for frustrum
		// -> Line drawing connects 2 verts at a time
		// -> Use vert-pairs to define all edges of frustrum
		const vert_xyz = [
			[origin, tl_max],
			[origin, tr_max],
			[origin, br_max],
			[origin, bl_max],

			[tl_max, tr_max],
			[tr_max, br_max],
			[br_max, bl_max],
			[bl_max, tl_max],

			[tl_min, tr_min],
			[tr_min, br_min],
			[br_min, bl_min],
			[bl_min, tl_min],
      	];

		return vert_xyz;
	}

	// ................................................................................................................

}