/* Contains code for handling 3D camera control */

// For convenience
const HALF_PI = Math.PI * 0.5;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

class Orbit_Camera {

	/*
	This class implements the UI for having 'orbit' control of a 3D scene (rendered on a canvas)
	Orbiting behavior is trying to mimic camera control in the 3D modeling software Blender
	*/

	// ................................................................................................................

	constructor(initial_distance = 1, world_up = [0, 1, 0], world_right = [1, 0, 0]) {

		// Record initial state so we can restore it on reset
		this._initial_distance = initial_distance;
		
		// Store world vectors & initialize camera vectors
		this.distance = initial_distance;
		this.worldvecs = {up: VEC3.norm([...world_up]), right: VEC3.norm([...world_right])};
		this.camvecs = {up: null, right: null, position_norm: null, origin: null};
		this.reset();
		
		// Config
		this.CONFIG = {
			zoom_sensitivity: 0.95,
			zoom_min: 0.05,
			zoom_max: 500,
			orbit_sensitivity: 0.005,
			shift_sensitivity: 0.005,
		}

		// Storage for 'orbit canvas' object (to be bound-to later)
		this._canvas = null;
	}

	bind_to_canvas = (orbit_canvas, render_func) => this._canvas = new Orbit_Canvas(orbit_canvas, this, render_func);

	// ................................................................................................................

	reset = () => {
		
		// Reset origin xyz (i.e. reset panning) & camera orientation
		this.camvecs.origin = [0, 0, 0];
		this.camvecs.up = [...this.worldvecs.up];
		this.camvecs.right = [...this.worldvecs.right];

		// Reset camera positioning
		const camforward = VEC3.norm_cross(this.camvecs.up, this.camvecs.right);
		this.camvecs.position_norm = VEC3.subtract(this.camvecs.origin, camforward);
		
		return;
	}

	// ................................................................................................................

	set_world_axes = (world_up = null, world_right = null) => {
		/* Helper used to modify world up/right axes */
		if (world_up != null) this.worldvecs.up = [...world_up];
		if (world_right != null) this.worldvecs.right = [...world_right];
	}

	// ................................................................................................................

	snap_to_axis = (snap_x = false, snap_y = false, snap_z = false, invert = false) => {

		const value = invert ? -1 : 1;
	    if (snap_x) {
			this.camvecs.position_norm = [value, 0, 0];
			this.camvecs.up = [0, 1, 0];
			this.camvecs.right = [0, 0, -value];
		} else if(snap_y) {
			this.camvecs.position_norm = [0, value, 0];
			this.camvecs.up = [0, 0, -value];
			this.camvecs.right = [1, 0, 0];
		} else if (snap_z) {
			this.camvecs.position_norm = [0, 0, value];
			this.camvecs.up = [0, 1, 0];
			this.camvecs.right = [value, 0, 0];
		}

	}

	// ................................................................................................................

	translate = (dx, dy, dz) => {

		/*
		Function used to move the camera origin in space.
		Movement is based on 2D dragging motion (e.g. mouse movement), and moves the camera
		left/right & up/down according to the camera orientation in 3D space. So for example,
		movement left/right is not along the world x-axis, but instead along 3D axis that
		is to the right of the camera viewing axis
		*/

		// Figure out the current (correct) up & right vectors, based on camera position
		const delta_up = this.camvecs.up.map(value => dy * value * this.CONFIG.shift_sensitivity);
		const delta_right = this.camvecs.right.map(value => -dx * value * this.CONFIG.shift_sensitivity);
		const delta_forward = this.camvecs.position_norm.map(value => -dz * value * this.CONFIG.shift_sensitivity);

		// Update origin offsets
		const translation_xyz = VEC3.add(VEC3.add(delta_up, delta_right), delta_forward);
		this.camvecs.origin = VEC3.add(this.camvecs.origin, translation_xyz);

		return;
	}

	// ................................................................................................................

	rotate = (dx, dy) => {

		/*
		Function used to rotate/orbit the camera, according to some amount of x/y dragging.
		This is a best-guess at mimicking the way Blender (the 3D modeling software) handles camera rotation.
		*/
        
		// Get rotation amounts
		const rot_angle_x_rad = -dx * this.CONFIG.orbit_sensitivity;
		const rot_angle_y_rad = -dy * this.CONFIG.orbit_sensitivity;

		// First rotate camera up/down around existing camera right axis,
		// then rotate camera position & right-axis left/right around world up axis
		// -> This gives nice behavior, even at [0, +/-1, 0] poles
		let rot_pos = this.camvecs.position_norm;
		rot_pos = rotate_axis_angle(rot_pos, this.camvecs.right, rot_angle_y_rad);
		rot_pos = rotate_axis_angle(rot_pos, this.worldvecs.up, rot_angle_x_rad);
		const new_right_axis = rotate_axis_angle(this.camvecs.right, this.worldvecs.up, rot_angle_x_rad);
		
		// // Update camera vectors
		const new_forward_axis = VEC3.subtract([0,0,0], rot_pos);
		this.camvecs.position_norm = VEC3.norm(rot_pos);
        this.camvecs.right = VEC3.norm(new_right_axis);
		this.camvecs.up = VEC3.norm_cross(new_right_axis, new_forward_axis);

        return;
	}

	// ................................................................................................................
	
	zoom = (zoom_delta) => {

		// Change distance of camera to create zooming effect
		const is_zoom_out = Math.sign(zoom_delta) > 0;
		this.distance *= is_zoom_out ? this.CONFIG.zoom_sensitivity : (1.0 / this.CONFIG.zoom_sensitivity);
		this.distance = clamp(this.distance, this.CONFIG.zoom_min, this.CONFIG.zoom_max);

		return;
	}

	// ...............................................................................................................

    get_world_to_view_mat4 = () => {

        /* Returns a matrix that maps positions in world space to view space (i.e. in front of camera) */

		// Scale normalized position into distance/zoomed position
		const scaled_xyz = this.camvecs.position_norm.map(xyz => xyz * this.distance);
		const world_xyz = VEC3.add(scaled_xyz, this.camvecs.origin);

		const lookat_mat = MAT4.look_at(world_xyz, this.camvecs.origin, this.camvecs.up);
		const view_matrix = MAT4.inverse(lookat_mat);

		return view_matrix;
    }

	// ................................................................................................................

    get_view_to_clipspace_mat4 = (fov_rad, display_ar = 1.0, use_orthographic_camera = false) => {
        
        /* Gives a matrix that maps view-space coords to clipspace (-1, +1) coords */

		// For clarity, define render distances based on how much we can zoom in
		const far_dist = this.CONFIG.zoom_max * 2;
		const near_dist = this.CONFIG.zoom_min * 0.25;

		// Build orthographic or perspective matrix, depending on configs
		let proj_matrix;
		if (use_orthographic_camera) {
			// Calculate 'zoom' effect due to movement of the origin & camera placement
			// -> Without this, ortho view won't change size when we move toward/away from the scene
			const movement_zoom = VEC3.dot(this.camvecs.origin, this.camvecs.position_norm);
			const zoom = (this.distance + movement_zoom) * 0.5;
			const ar_zoom = zoom * display_ar;
			const [l,r,b,t] = [-ar_zoom, ar_zoom, -zoom, zoom];
			proj_matrix = MAT4.orthographic(l,r,b,t, -far_dist, far_dist)
		} else {
			proj_matrix = MAT4.perspective(fov_rad, display_ar, near_dist, far_dist);
		}

		return proj_matrix;
    }

}


// --------------------------------------------------------------------------------------------------------------------


class Orbit_Canvas {

	/*
    Class used to handle updating of a html canvas element
    Based on an Orbit_Camera instance. Any time the camera is changed,
    a canvas rendering function is called.

	This class mainly helps to keep DOM/event code separate from camera control code
    */
   
	// ...............................................................................................................

    constructor(canvas_ref, orbit_camera_ref, render_func) {

		// Make sure the canvas_ref is a DOM element (can be given as id string or the element itself)
		if (typeof(canvas_ref) === "string") {
			canvas_ref = document.getElementById(canvas_ref);
		}

		// Awkward type checking, want to make sure we're taking the right type of input
		console.assert(orbit_camera_ref.constructor.name == Orbit_Camera.name, "Must use Orbit_Camera on canvas!")

		// Store inputs for re-use
		this.canvas = canvas_ref;
		this.orbitcam = orbit_camera_ref;
		this.render_func = render_func;

		// Storage for keeping track of click state
		this.is_active = false;
		this._is_middle_click = false;

		// Attach event listeners
		canvas_ref.addEventListener("pointerdown", this._on_click);
		canvas_ref.addEventListener("pointerup", this._on_finish)
		canvas_ref.addEventListener("pointercancel", this._on_finish);
		canvas_ref.addEventListener("pointermove", this._on_move);
		canvas_ref.addEventListener("wheel", this._on_zoom, {passive: false});
		canvas_ref.addEventListener("dblclick", this._on_reset);
    }

    _on_reset = (event) => {

        // Defer to camera controls for resetting
        this.orbitcam.reset();
		this.render_func();

        return;
    }

    _on_click = (event) => {

        // Record canvas as pointer target
        // -> Dragging off canvas doesn't cause lose of focus
        // -> Need to manually release the capture when pointer is released
        const pt_id = event.pointerId;
        this.canvas.setPointerCapture(pt_id);
        this.is_active = true;
		this._is_middle_click = event.which === 2;

        return;
    }

    _on_finish = (event) => {

        const pt_id = event.pointerId;
        this.canvas.releasePointerCapture(pt_id);
        this.is_active = false;
		this._is_middle_click = false;

        return;
    }
    
    _on_move = (event) => {

        // Ignore movement unless we're dragging
        if (!this.is_active) return;

		// Switch between panning & orbiting
		const is_shift = event.shiftKey || this._is_middle_click;
		if (is_shift) {
			this.orbitcam.translate(event.movementX, event.movementY, 0);
		} else {
			this.orbitcam.rotate(event.movementX, event.movementY);
		}
        
		this.render_func();

        return;
    }

    _on_zoom = (event) => {

        // Prevent page scroll
        event.preventDefault();
		event.stopPropagation();

		// Use wheel to zoom (add pinch-zoom support at some point?)
        this.orbitcam.zoom(event.wheelDeltaY);
        this.render_func();

        return;
    }
}