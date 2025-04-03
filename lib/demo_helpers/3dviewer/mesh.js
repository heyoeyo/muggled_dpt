
/*
Contains code used to produce a dense plane mesh for use in 3D rendering, where the dense vertices of the mesh
can be displaced to produce a 3D surface based on a depth prediction.
*/

function make_plane_mesh(width, height, num_faces = 250000) {

	/*
	Generates plane mesh data needed for rendering, along with some info about the mesh.
	The vertex data only includes xy coordinates, as the z coords. are meant to be derived
	from a depth prediction. The face data consists entirely of triangles (no quads).

	Returns:
		[vertex_xys_list, face_order_list, mesh_info_dict]
	
	Vertex xys look like: [[1.0, 1.0], [1.0, 0.9], [1.0, 0.8], ...]
	Face data looks like: [[0,1,2], [0,2,3], [2,4,5], [2,5,6], ...]
	*/

	// Prevent excessively high face counts (which will lock up the browser)
	const [min_faces, max_faces] = [2, 5_000_000];
	const target_face_count = Math.ceil(Math.max(Math.min(num_faces, max_faces), min_faces));
	const target_vert_count = Math.round(0.5 * target_face_count + Math.sqrt(target_face_count));

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
	const face_order_list = Array(total_num_faces);
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

	const mesh_info = {
		total_num_faces: face_order_list.length,
		target_face_count: target_face_count,
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
	const max_x_jitter = jitter_pct * mesh_info.x_vert_step * 0.5;
	const max_y_jitter = jitter_pct * mesh_info.y_vert_step * 0.5;
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