/*
Contains code used to load texture data (both image/rgb & depth data)
onto the GPU for rendering. This code is written with support for
modifying the texture data in-place for use with videos.
*/

class TextureData {

	/* Helper class used to manage/update the image & depth texture data for rendering */

	constructor(webgl_context, shader_texture_varnames) {
		this._gl = webgl_context;
		this._texturevars = shader_texture_varnames;

		this._image_texture = null;
		this._depth_texture = null;
		this._image_texture_unit = null;
		this._depth_texture_unit = null;

		this._image_wh = [0,0];
		this._depth_wh = [0,0];
		this._unit_offset_image = 0;
		this._unit_offset_depth = 1;
	}

	// ................................................................................................................

	allocate = (gl_program, image_wh, depth_wh, num_mipmaps=3) => {

		/*
		Helper used to set up storage for rendering textures.
		This is meant to be called once the width/height of the texture data is known
		*/

		// For convenience
		const gl = this._gl;

		// Re-allocate image/depth data if it changes size
		// - These are allocated independent of the shaders/program being used!
		// - Only need to re-allocate if storage requirements change (e.g. image size changes)
		if (image_wh[0] != this._image_wh[0] || image_wh[1] != this._image_wh[1]) {
			_clear_texture_storage(gl, this._image_texture);
			this._image_texture = _allocate_texture_storage(gl, image_wh, this._unit_offset_image, num_mipmaps);
			this._image_wh = image_wh;
		}
		if (depth_wh[0] != this._depth_wh[0] || depth_wh[1] != this._depth_wh[1]) {
			_clear_texture_storage(gl, this._depth_texture);
			this._depth_texture = _allocate_texture_storage(gl, depth_wh, this._unit_offset_depth, num_mipmaps);
			this._depth_wh = depth_wh;
		}
		
		// Attach textures to given program
		this._image_texture_unit = _bind_texture(
			gl, gl_program, this._image_texture, this._unit_offset_image, this._texturevars.image,
		);
		this._depth_texture_unit = _bind_texture(
			gl, gl_program, this._depth_texture, this._unit_offset_depth, this._texturevars.depth,
		);

		return this;
	}

	// ................................................................................................................
	
	update = (image_bitmap, depth_bitmap) => {

		/* Helper used to load new image/depth data for rendering */

		// For clarity
		const gl = this._gl;
		const texture_target = WebGL2RenderingContext.TEXTURE_2D;
		const level = 0;
		const x1 = 0;
		const y1 = 0;
		const format = WebGL2RenderingContext.RGB;
		const dtype = WebGL2RenderingContext.UNSIGNED_BYTE;

		// Update image texture
		const wi = image_bitmap.width;
		const hi = image_bitmap.height;
		gl.activeTexture(this._image_texture_unit);
		gl.bindTexture(texture_target, this._image_texture);
		gl.texSubImage2D(texture_target, level, x1, y1, wi, hi, format, dtype, image_bitmap);
		gl.generateMipmap(texture_target);

		// Update depth texture
		const wd = depth_bitmap.width;
		const hd = depth_bitmap.height;
		gl.activeTexture(this._depth_texture_unit);
		gl.bindTexture(texture_target, this._depth_texture);
		gl.texSubImage2D(texture_target, level, x1, y1, wd, hd, format, dtype, depth_bitmap);
		gl.generateMipmap(texture_target);

		return this;
	}
}

// ....................................................................................................................

function _allocate_texture_storage(
	gl_ref,
	texture_wh,
	texture_unit_offset,
	num_mipmaps=3,
	format=WebGL2RenderingContext.RGB8,
	target=WebGL2RenderingContext.TEXTURE_2D,
) {
	/* Helper used to create storage for texture data on the GPU. Only needs to be done once for a fixed image size */

	// For clarity
	const texture_unit = gl_ref.TEXTURE0 + texture_unit_offset;
	const webgl_texture = gl_ref.createTexture();
	
	// Set up texture
	gl_ref.activeTexture(texture_unit);
	gl_ref.bindTexture(gl_ref.TEXTURE_2D, webgl_texture);
	gl_ref.texParameteri(gl_ref.TEXTURE_2D, gl_ref.TEXTURE_WRAP_S, gl_ref.CLAMP_TO_EDGE);
	gl_ref.texParameteri(gl_ref.TEXTURE_2D, gl_ref.TEXTURE_WRAP_T, gl_ref.CLAMP_TO_EDGE);
	gl_ref.texParameteri(gl_ref.TEXTURE_2D, gl_ref.TEXTURE_MIN_FILTER, gl_ref.LINEAR_MIPMAP_LINEAR);
	gl_ref.texParameteri(gl_ref.TEXTURE_2D, gl_ref.TEXTURE_MAG_FILTER, gl_ref.LINEAR);
	gl_ref.texStorage2D(target, num_mipmaps, format, texture_wh[0], texture_wh[1]);

	return webgl_texture;
}

// ....................................................................................................................

function _bind_texture(gl_ref, gl_program, webgl_texture, texture_unit_offset, texture_shader_name) {

	/* Helper used to bind texture data to a specific program. Needs to be called if program changes */

	// Set up texture
	const texture_unit = gl_ref.TEXTURE0 + texture_unit_offset;
	gl_ref.activeTexture(texture_unit);
	gl_ref.bindTexture(gl_ref.TEXTURE_2D, webgl_texture);

	// Set up uniform naming inside of shader
	const texture_shader_loc = gl_ref.getUniformLocation(gl_program, texture_shader_name);
	gl_ref.uniform1i(texture_shader_loc, texture_unit_offset);

	return texture_unit;
}

// ....................................................................................................................

function _clear_texture_storage(gl_ref, webgl_texture) {
	/* Helper used to delete texture, only if it exists */
	if (webgl_texture != null) {
		gl_ref.deleteTexture(webgl_texture);
	}
}