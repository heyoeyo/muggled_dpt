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

		// Special storage, only used if we try to 'save' texture data which requires creating a frame buffer!
		this._framebuffer = null;
	}

	// ................................................................................................................

	allocate = (image_wh, depth_wh, num_mipmaps=3) => {

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

		return this;
	}

	// ................................................................................................................

	bind_gl_program = (gl_program) => {

		/* Helper used to bind existing texture data to a different gl program/shaders */

		this._image_texture_unit = _bind_texture(
			this._gl, gl_program, this._image_texture, this._unit_offset_image, this._texturevars.image,
		);
		this._depth_texture_unit = _bind_texture(
			this._gl, gl_program, this._depth_texture, this._unit_offset_depth, this._texturevars.depth,
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

	// ................................................................................................................

	read_current_texture_data = () => {

		// Create/bind framebuffer, needed for reading out texture data
		const gl = this._gl;
		if (this._framebuffer === null) this._framebuffer = gl.createFramebuffer();

		// Read both textures
		const image_pixels = _read_texture_from_framebuffer(gl, this._framebuffer, this._image_texture, this._image_wh);
		const depth_pixels = _read_texture_from_framebuffer(gl, this._framebuffer, this._depth_texture, this._depth_wh);

		const image_data = new ImageData(image_pixels, this._image_wh[0], this._image_wh[1]);
		const depth_data = new ImageData(depth_pixels, this._depth_wh[0], this._depth_wh[1]);

		return [image_data, depth_data];
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

// ....................................................................................................................

function _read_texture_from_framebuffer(gl_ref, framebuffer_ref, texture_ref, texture_wh, num_channels=4) {

	/* Helper used to read out texture data from webgl. Returns a Uint8Array */

	gl_ref.bindFramebuffer(gl_ref.FRAMEBUFFER, framebuffer_ref);
	gl_ref.framebufferTexture2D(gl_ref.FRAMEBUFFER, gl_ref.COLOR_ATTACHMENT0, gl_ref.TEXTURE_2D, texture_ref, 0);
	if (gl_ref.checkFramebufferStatus(gl_ref.FRAMEBUFFER) !== gl_ref.FRAMEBUFFER_COMPLETE) {
		console.error("Framebuffer is not complete");
	}

	// Allocate storage for pixel data, according to image size
	const [width, height] = texture_wh;
	const num_image_pixels = num_channels * width * height;
	const pixel_data = new Uint8ClampedArray(num_image_pixels);

	// Read texture data into array
	const include_alpha = (num_channels === 4);
	const format = include_alpha ? gl_ref.RGBA : gl_ref.RGB;
	const dtype = WebGLRenderingContext.UNSIGNED_BYTE;
	gl_ref.readPixels(0, 0, width, height, format, dtype, pixel_data);

	// Unbind framebuffer (otherwise all future rendering goes to this buffer, instead of the screen)
	gl_ref.bindFramebuffer(gl_ref.FRAMEBUFFER, null);

	return pixel_data;
}

// ....................................................................................................................

function save_image_data(save_name, image_data, save_ext="png", requires_y_flip=true) {
	/* Helper used to handle saving/downloading of image data */

	// Create HTML elements needed to handle rendering/saving
	const temp_canvas = document.createElement("canvas");
	const temp_ctx = temp_canvas.getContext("2d");
	const temp_a = document.createElement("a");
	temp_a.download = `${save_name}.${save_ext}`;

	// Use try-catch to make sure we properly clean up temp. elements, regardless of errors
	try {
		// Draw image to canvas, needed to encode image data (e.g. as png/jpg)
		temp_canvas.width = image_data.width;
		temp_canvas.height = image_data.height;
		temp_ctx.putImageData(image_data, 0, 0);
		if (requires_y_flip) {
			temp_ctx.scale(1, -1);
			temp_ctx.drawImage(temp_canvas, 0, -temp_canvas.height);
		}

		// Encode & save image data
		const content_type = `image/${save_ext}`;
		temp_canvas.toBlob(blob => {
		  const temp_url = URL.createObjectURL(blob);
		  temp_a.href = temp_url;
		  temp_a.click();
		  URL.revokeObjectURL(temp_url);
		}, content_type);

	} catch {
		console.warn(`Error saving image data: ${temp_a.download}`);
	}

	// Clean up
	temp_a.remove();
	temp_canvas.remove();

	return;
}