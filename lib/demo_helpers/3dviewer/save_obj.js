
function create_obj_string(vertex_xyz_list, vertex_uv_list, face_indexing_list) {
	/*
	Function used to construct a .obj (string) representation of the given
	3d model data, which can then be rendered in other programs (like Blender)
	*/

	const num_vertices = vertex_xyz_list.length;
	const num_faces = face_indexing_list.length;
	const num_verts_per_face = face_indexing_list[0].length;
	const obj_strs_list = [
		"# Made with MuggledDPT: https://github.com/heyoeyo/muggled_dpt",
		`# ${num_vertices} vertices  |  ${num_faces} faces  |  ${num_verts_per_face} verts per face`,
		"o depth_prediction",
	];

	// Add vertex position data
	for(const [vx, vy, vz] of vertex_xyz_list) {
		obj_strs_list.push(`v ${vx} ${vy} ${vz}`);
	}

	// Add vertex UVs
	for(const [vu, vv] of vertex_uv_list) {
		obj_strs_list.push(`vt ${vu} ${vv}`);
	}

	// Add face indexing (obj counts indexing starting at 1!)
	let offset_idxs;
	for(let vert_idxs of face_indexing_list) {
		offset_idxs = vert_idxs.map(i => 1 + i);
		obj_strs_list.push(`f ${offset_idxs.map(i => `${i}/${i}`).join(" ")}`);
	}

	return obj_strs_list.join("\n");
}

// ....................................................................................................................

function save_obj_data(save_name, obj_save_str) {
	/* Helper used to handle saving/downloading of .obj (text) data */

	// Create HTML element needed to trigger download
	const temp_a = document.createElement("a");
	temp_a.download = `${save_name}.obj`;

	try{
		// Convert .obj to text for saving
		const blob = new Blob([obj_save_str], { type: "text/plain" });
		const temp_url = URL.createObjectURL(blob);
		temp_a.href = temp_url;
		temp_a.click();
		URL.revokeObjectURL(temp_url);
	} catch {
		console.warn(`Error saving obj: ${temp_a.download}`);
	}

	// Clean up
	temp_a.remove();

	return;
}