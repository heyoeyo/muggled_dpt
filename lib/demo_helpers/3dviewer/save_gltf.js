
async function create_glb_binary(vertex_xyz_list, vertex_uv_list, face_indexing_list, image_data, image_ext="png") {

  /*
  Function used to build gltf binary data (e.g. as a single .glb file)
  Handles both the construction of the JSON structure describing the model,
  as well as packing together the mesh/texture data

  Returns: glb_byte_data (as a Uint8Array)
  */

  // For convenience
  pad_to_4bytes = (num) =>  Math.ceil(num/4) * 4;

  // Find vertex xyz min/max
  let [min_x, min_y, min_z] = [1E6, 1E6, 1E6];
  let [max_x, max_y, max_z] = [-1E6, -1E6, -1E6];
  for(const [vx, vy, vz] of vertex_xyz_list) {
    min_x = Math.min(vx, min_x);
    min_y = Math.min(vy, min_y);
    min_z = Math.min(vz, min_z);
    max_x = Math.max(vx, max_x);
    max_y = Math.max(vy, max_y);
    max_z = Math.max(vz, max_z);
  }
  const min_xyz = [min_x, min_y, min_z];
  const max_xyz = [max_x, max_y, max_z];

  // Convert to flat arrays
  const image_mimetype = `image/${image_ext}`
  const xyz_array = new Float32Array(vertex_xyz_list.flat());
  const uv_array = new Float32Array(vertex_uv_list.flat());
  const faces_array = new Uint32Array(face_indexing_list.flat());
  const texture_arraybuffer = await _image_data_to_arraybuffer(image_data, image_mimetype);

  // Get byte offsets, assuming we stack in xyz/uv/indices/texture order
  const xyz_offset = 0;
  const uv_offset = xyz_offset + xyz_array.byteLength;
  const faces_offset = uv_offset + uv_array.byteLength;
  const texture_offset = faces_offset + faces_array.byteLength;
  const model_byte_length = pad_to_4bytes(
    xyz_array.byteLength + uv_array.byteLength + faces_array.byteLength + texture_arraybuffer.byteLength
  );
  
  // For clarity
  const [type_uint32, type_float32] = [5125, 5126];
  const [targ_abuf, targ_ebuf] = [34962, 34963];
  const num_verts = vertex_xyz_list.length;
  const num_uvs = vertex_uv_list.length;
  const num_indices = faces_array.length;

  // Decide on rendering mode (e.g. triangles vs. points vs. lines)
  // See: https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/MeshPrimitiveModes/README.md#structure
  const num_verts_per_face = face_indexing_list[0].length;
  console.assert(num_verts_per_face > 0 && num_verts_per_face < 4, "Faces must have 1 to 3 vertices only!");
  const vpf_to_mode_lut = {1: 0, 2: 1, 3: 4};
  const mode = vpf_to_mode_lut[num_verts_per_face];

  // Create gltf JSON structure describing model data
  const glb_json = {
    asset: { version: "2.0" },
    buffers: [{ byteLength: model_byte_length }],
    bufferViews: [
      {buffer: 0, byteOffset: xyz_offset,     byteLength: xyz_array.byteLength,   target: targ_abuf},
      {buffer: 0, byteOffset: uv_offset,      byteLength: uv_array.byteLength,    target: targ_abuf},
      {buffer: 0, byteOffset: faces_offset,   byteLength: faces_array.byteLength, target: targ_ebuf},
      {buffer: 0, byteOffset: texture_offset, byteLength: texture_arraybuffer.byteLength},
    ],
    accessors: [
      {bufferView: 0, componentType: type_float32, count: num_verts,   type: "VEC3", max: max_xyz, min: min_xyz},
      {bufferView: 1, componentType: type_float32, count: num_uvs,     type: "VEC2"},
      {bufferView: 2, componentType: type_uint32,  count: num_indices, type: "SCALAR"},
    ],
    images: [{bufferView: 3, mimeType: image_mimetype}],
    textures: [{source: 0}], // Means 0th entry of 'images' list
    materials: [{
        extensions: {KHR_materials_unlit: {}},
        pbrMetallicRoughness: {baseColorTexture: {index: 0}}, // Means 0th entry of 'textures' list
    }],
    extensionsUsed: ["KHR_materials_unlit"],
    meshes: [{
      primitives: [{
        attributes: {POSITION: 0, TEXCOORD_0: 1},
        indices: 2,
        material: 0,
        mode: mode,
      }]
    }],
    nodes: [{mesh: 0, name: "depth_prediction"}],
    scenes: [{nodes: [0]}],
    scene: 0,
  };
  const json_byte_data = new TextEncoder().encode(JSON.stringify(glb_json));
  const json_byte_length = pad_to_4bytes(json_byte_data.length);
  console.log("GLTF JSON Data:", glb_json)
  
  // For clarity
  const version_code = 2;
  const gltf_hexcode = 0x46546C67;
  const json_hexcode = 0x4E4F534A;
  const data_hexcode = 0x004E4942;
  
  // Compute total length of all data (headers + json + model data)
  const total_header_length = (3 + 2 + 2) * 4;
  const total_glb_byte_length = total_header_length + json_byte_length + model_byte_length;

  // Figure out header data/sizing
  const glb_header_offset = 0;
  const glb_header_list = [gltf_hexcode, version_code, total_glb_byte_length];
  const json_header_offset = 12;
  const json_header_list = [json_byte_length, json_hexcode];
  const json_byte_offset = json_header_offset + json_header_list.length * 4;
  const model_header_offset = 20 + json_byte_length;
  const model_header_list = [model_byte_length, data_hexcode];
  const model_byte_offset = model_header_offset + model_header_list.length * 4;
  
  // Allocate storage for final output binary data
  const glb_byte_data = new Uint8Array(total_glb_byte_length);
  const glb_dataview = new DataView(glb_byte_data.buffer);

  // Write headers
  write_header = (offset, header_list, is_little_endian=true) => {
      for(let i=0; i<header_list.length; i++) {
          glb_dataview.setUint32(offset + 4*i, header_list[i], is_little_endian);
      }
  }
  write_header(glb_header_offset, glb_header_list);
  write_header(json_header_offset, json_header_list);
  write_header(model_header_offset, model_header_list);

  // Write JSON data (with ' ' padding at the end, otherwise we get parsing errors)
  const padded_json_array = new Uint8Array(json_byte_length);
  padded_json_array.set([32,32,32,32], json_byte_length - 4)
  padded_json_array.set(json_byte_data);
  glb_byte_data.set(padded_json_array, json_byte_offset);

  // Write model data
  const padded_model_data = new Uint8Array(model_byte_length);
  padded_model_data.set(new Uint8Array(xyz_array.buffer), xyz_offset);
  padded_model_data.set(new Uint8Array(uv_array.buffer), uv_offset);
  padded_model_data.set(new Uint8Array(faces_array.buffer), faces_offset);
  padded_model_data.set(new Uint8Array(texture_arraybuffer), texture_offset);
  glb_byte_data.set(padded_model_data, model_byte_offset);

  return glb_byte_data;
}

// ....................................................................................................................

function save_glb_data(save_name, glb_binary_data) {
    /* Helper used to handle saving/downloading of .glb (Uint8Array) data */

    // Create HTML element needed to trigger download
    const temp_a = document.createElement("a");
    temp_a.download = `${save_name}.glb`;

    try{
        // Convert .obj to text for saving
        const blob = new Blob([glb_binary_data], { type: "model/gltf-binary" });
        const temp_url = URL.createObjectURL(blob);
        temp_a.href = temp_url;
        temp_a.click();
        URL.revokeObjectURL(temp_url);
    } catch {
        console.warn(`Error saving: ${temp_a.download}`);
    }

    // Clean up
    temp_a.remove();

    return;
}

// ....................................................................................................................

function _image_data_to_blob(image_data, data_type="image/png") {

  /* Helper used to map 'ImageData' to a blob, which provides jpg/png encoding */

  const temp_canvas = document.createElement("canvas");
  const ctx = temp_canvas.getContext("2d");

  temp_canvas.width = image_data.width;
  temp_canvas.height = image_data.height;
  ctx.putImageData(image_data, 0, 0);

  return new Promise((resolve) => {
    temp_canvas.toBlob(resolve, data_type);
    temp_canvas.remove();
  });
}

async function _image_data_to_arraybuffer(image_data, data_type="image/png"){
  /* Helper used to convert 'ImageData' to an encoded (png/jpg) buffer of bytes */
  const blob = await _image_data_to_blob(image_data, data_type);
  return await blob.arrayBuffer();
}