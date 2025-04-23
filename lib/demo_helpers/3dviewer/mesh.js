
/*
Contains code used to produce a dense plane mesh for use in 3D rendering, where the dense vertices of the mesh
can be displaced to produce a 3D surface based on a depth prediction.
*/

class MeshData {

	/* Helper class used to manage vertex & face indexing data needed for rendering */

	constructor(webgl_context, shaderdefs_attrs) {

		// Hold inputs for re-use in data init/rendering
		this._gl = webgl_context;
		this._attrvars = shaderdefs_attrs;

		// Storage for holding mesh data (mostly for debugging)
		this._vert_xys = [];
		this._face_indexing = [];
		this._plane_mesh_info = {};
		this._num_elements = 0;

		this._vert_attr_loc = null;
		this._vao = this._gl.createVertexArray();

		// Buffer for holding vertex xy positions array
		this._vert_buffer_type = WebGL2RenderingContext.ARRAY_BUFFER;
		this._vert_buffer = this._gl.createBuffer();

		// Buffer for holding face indexing array
		this._face_buffer_type = WebGL2RenderingContext.ELEMENT_ARRAY_BUFFER;
		this._face_buffer = this._gl.createBuffer();
	}

	// ................................................................................................................

	bind_gl_program = (gl_program) => {

		/* Function used to link mesh data to a webgl program. Needs to be called if the program changes */

		// For clarity
		const gl = this._gl;
		gl.useProgram(gl_program);

		// Get reference to attribute variables defined in shader code
		this._vert_attr_loc = gl.getAttribLocation(gl_program, this._attrvars.vertex_xy);

		// Attach vertex array object to new program
		gl.bindVertexArray(this._vao);
		gl.enableVertexAttribArray(this._vert_attr_loc);

		// Re-load existing mesh data into new program
		if (this._num_elements > 0) {
			this._bind_mesh_data(this._vert_xys, this._face_indexing, this._vert_attr_loc);
		}

		return this;
	}

	// ................................................................................................................

	update = (
		mesh_wh, target_num_faces, vertex_jitter_pct, draw_mode=WebGL2RenderingContext.TRIANGLES,
	) => {
		/* Function used to update mesh data used for rendering. This isn't meant to be called often! */

		let [vert_xys, face_list, info] = this._make_mesh_data(mesh_wh, target_num_faces, vertex_jitter_pct, draw_mode);

		// Store resulting mesh data (mostly needed for saving 3d model)
		this._num_elements = this._bind_mesh_data(vert_xys, face_list, this._vert_attr_loc);
		this._vert_xys = vert_xys;
		this._face_indexing = face_list;
		this._plane_mesh_info = info;

		return;
	}

	// ................................................................................................................

	draw_elements = (draw_mode) => {
		/* Draw whatever is currently in the element array buffer (e.g face indexing data) */
		const offset = 0;
		this._gl.drawElements(draw_mode, this._num_elements, WebGL2RenderingContext.UNSIGNED_INT, offset);
	}

	// ................................................................................................................

	_bind_mesh_data = (vertex_xys, face_indexing_list, vertex_attr_location) => {
		/*
		Function used to bind mesh data to the vertex position & face indexing buffers.
		This needs to be done whenever the mesh data changes (e.g. when changing mesh density or jitter),
		or if the gl program is changed (i.e. using different shaders on the same geometry).
		*/

		// For clarity
		const gl = this._gl;
		const num_dimensions = vertex_xys[0].length;
		const num_faces = face_indexing_list.length;
		const num_verts_per_face = face_indexing_list[0].length;
		const num_elements = num_faces * num_verts_per_face;
		const usage_type = gl.STATIC_DRAW;

		// Specify how attribute data (xy coords) is formatted within the buffer for reading in vertex shader
		const attr_size = num_dimensions;
		const attr_type = WebGL2RenderingContext.FLOAT;
		const normalize = false;
		const buffer_stride = 0;
		const buffer_offset = 0;
		gl.vertexAttribPointer(vertex_attr_location, attr_size, attr_type, normalize, buffer_stride, buffer_offset);

		// Bind vertex buffer & data to current gl program
		gl.bindBuffer(this._vert_buffer_type, this._vert_buffer);
		gl.bufferData(this._vert_buffer_type, new Float32Array(vertex_xys.flat()), usage_type);

		// Bind face indexing data to current gl program
		gl.bindBuffer(this._face_buffer_type, this._face_buffer);
		gl.bufferData(this._face_buffer_type, new Uint32Array(face_indexing_list.flat()), usage_type);

		return num_elements;
	}

	// ................................................................................................................

	_make_mesh_data = (mesh_wh, target_num_faces, vertex_jitter_pct, draw_mode=WebGL2RenderingContext.TRIANGLES) => {

		/* Helper function used to re-build mesh data */

		// Create plane mesh
		const [mesh_w, mesh_h] = mesh_wh;
		let [vertex_xys, face_indexing_list, mesh_info] = _make_plane_mesh(
			mesh_w, mesh_h, target_num_faces, draw_mode,
		);

		// Add jitter to xys
		if (vertex_jitter_pct > 0.0001) {
			vertex_xys = apply_mesh_jitter(vertex_xys, mesh_info, vertex_jitter_pct);
		}

		return [vertex_xys, face_indexing_list, mesh_info];
	}

	// ................................................................................................................

	read_current_mesh_data = () => [this._vert_xys, this._face_indexing];
}

// ...................................................................................................................

function _make_plane_mesh(width, height, target_face_count=250000, draw_mode=WebGL2RenderingContext.TRIANGLES) {

	/*
	Generates plane mesh data needed for rendering, along with some info about the mesh.
	The vertex data only includes xy coordinates, as the z coords. are meant to be derived
	from a depth prediction. The face data consists entirely of triangles (no quads).

	Returns:
		[vertex_xys_list, face_order_list, mesh_info_dict]
	
	Vertex xys look like: [[1.0, 1.0], [1.0, 0.9], [1.0, 0.8], ...]
	Face data looks like: [[0,1,2], [0,2,3], [2,4,5], [2,5,6], ...]
	*/

	// Make sure we have at least 2 faces to form a rectangle
	const clamp_target_faces = Math.max(target_face_count, 2);
	const target_vert_count = Math.round(0.5 * clamp_target_faces + Math.sqrt(clamp_target_faces));

	// Figure out how many vertices to assign to x/y grid
	const aspect_ratio = width / height;
	const raw_num_x = Math.sqrt(target_vert_count * aspect_ratio);
	const raw_num_y = raw_num_x / aspect_ratio;
	const num_x_verts = Math.max(Math.round(raw_num_x), 2);
	const num_y_verts = Math.max(Math.round(raw_num_y), 2);
	const total_num_verts = num_x_verts * num_y_verts;
	const [max_col_idx, max_row_idx] = [num_x_verts - 1, num_y_verts - 1];

	// Compute x/y coordinate steps (e.g. lerp steps from -1 to +1)
	const [x_step, y_step] = [2.0 / max_col_idx, 2.0 / max_row_idx];
	const x_coords = Array.from({ length: num_x_verts }, (v, i) => i * x_step - 1.0);
	const y_coords = Array.from({ length: num_y_verts }, (v, i) => 1.0 - i * y_step);

	// Figure out how many faces we'll have so we can pre-allocate array data (for optimization)
	const total_num_faces = (total_num_verts - num_x_verts - num_y_verts + 1) * 2;
	let face_order_list = Array(total_num_faces);
	let face_idx = 0;

	// Generate vertex xy for full mesh (as 1D array) along with triangle ordering
	let v_idx, vr_idx, vb_idx, vbr_idx;
	const vert_xys_list = Array(total_num_verts);
	for(const [row_idx, y_val] of y_coords.entries()) {
		for(const [col_idx, x_val] of x_coords.entries()) {
			
			// Store vertex xy
			v_idx = col_idx + row_idx * num_x_verts;
			vert_xys_list[v_idx] = [x_val, y_val];
			
			// Skip forming triangles when we're at the far right/bottom edge of the mesh
			if (col_idx >= max_col_idx || row_idx >= max_row_idx) continue; 
			
			// Store triangle indexing (two tris per vertex)
			// -> Formed between vertices to the bottom/right of this vertex
			// -> Using counter-clockwise ordering to form faces!
			vr_idx = v_idx + 1;
			vb_idx = v_idx + num_x_verts;
			vbr_idx = vb_idx + 1;
			face_order_list[face_idx] = [v_idx, vb_idx, vbr_idx];
			face_order_list[face_idx + 1] = [v_idx, vbr_idx, vr_idx];
			face_idx += 2;
		}
	}

	// Alter faces if needed
	if (draw_mode === WebGL2RenderingContext.POINTS) {
		face_order_list = _face_indexing_as_points(vert_xys_list.length);
	} else if (draw_mode === WebGL2RenderingContext.LINES) {
		face_order_list = _triangles_to_wireframe(face_order_list);
	} else if(draw_mode != WebGL2RenderingContext.TRIANGLES) {
		// Warning if we get an unknown face type
		console.warn(`Unknown mesh face type: ${draw_mode}`);
	}

	const mesh_info = {
		total_num_faces: face_order_list.length,
		target_face_count: clamp_target_faces,
		total_num_verts: vert_xys_list.length,
		target_vert_count: target_vert_count,
		num_x_verts: num_x_verts,
		num_y_verts: num_y_verts,
		x_vert_step: x_step,
		y_vert_step: y_step,
	}

	return [vert_xys_list, face_order_list, mesh_info];
}

// ...................................................................................................................

function apply_mesh_jitter(vert_xys_list, mesh_info, jitter_pct) {

	/*
	Takes in mesh vertices and randomly nudges them
	- Assumes a plane mesh, with x/y values between -1 & +1 (boundary vertices aren't nudged)
	- Vertices are only displaced in a limited circular region surrounding their initial position
	*/

	// For clarity
	const reduction_factor = 0.9;
	const max_x_jitter = jitter_pct * mesh_info.x_vert_step * 0.5 * reduction_factor;
	const max_y_jitter = jitter_pct * mesh_info.y_vert_step * 0.5 * reduction_factor;
	const two_pi = Math.PI * 2.0;

	// Offset each vertex (radially) by a random amount
	for(let i = 0; i < vert_xys_list.length; i++) {
		if (Math.abs(vert_xys_list[i][0]) != 1 && Math.abs(vert_xys_list[i][1]) != 1) {
			const offset = Math.random();
			const angle = Math.random() * two_pi;
			vert_xys_list[i][0] += Math.cos(angle) * offset * max_x_jitter;
			vert_xys_list[i][1] += Math.sin(angle) * offset * max_y_jitter;
		}
	}

	return vert_xys_list;
}

// ...................................................................................................................

function _face_indexing_as_points(num_vertices) {

	/*
	Helper used to create 'points' based face indexing, where
	each face is just made of a single vertex.
	*/

	const new_faces_list = Array(num_vertices);
	for (let idx = 0; idx < num_vertices; idx++) {
		new_faces_list[idx] = [idx];
	}

	return new_faces_list;
}

// ...................................................................................................................

function _triangles_to_wireframe(face_indexing_list) {

	/*
	Helper used to convert triangle face indexing into a 'wireframe' format,
	meant to be rendered as webgl 'LINES'. Note, this implementation is very
	inefficient! It duplicates almost all line segments, it is only meant for debugging
	This function assumes the given face indexing is made of pairs of triangles forming
	quads, so that a wireframe can be made by breaking each triangle into 3 line segments.
	*/

	const num_lines = face_indexing_list.length * 3;
	const new_faces_list = Array(num_lines);
	let face, newidx;
	for (let idx = 0; idx < face_indexing_list.length; idx++) {
		face = face_indexing_list[idx];
		newidx = idx*3;
		new_faces_list[newidx+0] = [face[0], face[1]];
		new_faces_list[newidx+1] = [face[1], face[2]];
		new_faces_list[newidx+2] = [face[2], face[0]];
	}

	return new_faces_list;
}

// ...................................................................................................................

function filter_mesh_vertices(is_valid_list, vertex_xyz_list, vertex_uv_list, face_indexing_list) {

	/*
	Function used to prune mesh vertices (and corresponding faces) based on a
	'is valid' listing for each vertex. This is meant to be used with masking,
	so that parts of the mesh not included in the mask are removed from
	the actual 3D geometry.
	Returns: [new_xyz_list, new_uv_list, new_face_indexing_list]
	*/

	// Initialize outputs
	const new_xyz_list = Array();
	const new_uv_list = Array();
	const new_faces_list = Array();
	const old_to_new_idx_lut = new Map();

	// Generate re-indexed vertices (i.e. account for gaps introduced by deleting vertices)
	let next_vert_idx = 0;
	for(let i=0; i<vertex_xyz_list.length; i++) {
		if(is_valid_list[i]) {
			new_xyz_list.push(vertex_xyz_list[i]);
			new_uv_list.push(vertex_uv_list[i]);
			old_to_new_idx_lut.set(i, next_vert_idx);
			next_vert_idx += 1;
		}
	}

	// Re-index faces and remove any faces containing unused vertices
	let is_ok_face;
	for(let face of face_indexing_list) {
		is_ok_face = true;
		for(let vert_idx of face) {
			is_ok_face = is_valid_list[vert_idx];
			if (!is_ok_face) break;
		}
		if (is_ok_face) {
			new_faces_list.push(face.map(old_idx => old_to_new_idx_lut.get(old_idx)));
		}
	}

	return [new_xyz_list, new_uv_list, new_faces_list];
}