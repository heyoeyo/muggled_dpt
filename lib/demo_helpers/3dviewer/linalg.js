
/*
This script holds helper functions for performing vector and matrix operations
Much of the matrix related code comes from:
https://webgl2fundamentals.org/
*/

// Helper obj, holds 3D vector operations
const VEC3 = {

  norm: (v) => {
    const length = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    // make sure we don't divide by 0.
    if (length > 0.00001) {
      return [v[0] / length, v[1] / length, v[2] / length];
    } else {
      return [0, 0, 0];
    }
  },

  dot: (a,b) => (a[0]*b[0]) + (a[1]*b[1]) + (a[2]*b[2]),

  cross: (a, b) => {
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]];
  },

  norm_cross: (a, b) => VEC3.norm(VEC3.cross(a, b)),

  subtract: (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]],

  add: (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]],

  pointwise_multiply: (a, b) => [a[0]*b[0], a[1]*b[1], a[2]*b[2]],

  scalar_multiply: (vec, scalar) => [vec[0]*scalar, vec[1]*scalar, vec[2] * scalar],

  projection: (a, onto_b) => VEC3.scalar_multiply(onto_b, VEC3.dot(a, onto_b) / VEC3.dot(onto_b, onto_b)),
}

// ....................................................................................................................

function rotate_axis_angle(point, rot_axis_vector, rot_axis_angle_rad) {

  /*
  Rotate a point with an axis-angle vector (i.e. rotation axis + amount/angle to rotate)
  Uses Rodrigues' rotation formula:
  https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Statement
  */

  // Pre-calculate some quantities for convenience
  const cos = Math.cos(rot_axis_angle_rad);
  const sin = Math.sin(rot_axis_angle_rad);
  const a_cross_p = VEC3.cross(rot_axis_vector, point);
  const a_dot_p = VEC3.dot(rot_axis_vector, point);
  const scaled_a_dot_p = (1.0 - cos) * a_dot_p;

  // Calculate new point position using axis-angle rotation
  let point_term, cross_term, axis_term;
  const rotated_point = new Array(3);
  for (let k = 0; k < point.length; k++) {
    point_term = cos * point[k];
    cross_term = sin * a_cross_p[k];
    axis_term = scaled_a_dot_p * rot_axis_vector[k];
    rotated_point[k] = point_term + cross_term + axis_term;
  }

  return rotated_point;
}

// ....................................................................................................................

// Helper obj, holds 4D matrix operations
const MAT4 = {
  orthographic: function(left, right, bottom, top, near, far) {
    return [
      2 / (right - left), 0, 0, 0,
      0, 2 / (top - bottom), 0, 0,
      0, 0, 2 / (near - far), 0,
 
      (left + right) / (left - right),
      (bottom + top) / (bottom - top),
      (near + far) / (near - far),
      1,
    ];
	},
  
  perspective: function(fov_radians, aspect, near, far) {
    const f = Math.tan(0.5 * (Math.PI - fov_radians));
    const range_inv = 1.0 / (near - far);

    return [
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (near + far) * range_inv, -1,
      0, 0, near * far * range_inv * 2, 0,
    ];
  },

  rotate_x: function(angle_rad) {
    /* From: https://webglfundamentals.org/webgl/lessons/webgl-3d-orthographic.html */
    const cos = Math.cos(angle_rad);
    const sin = Math.sin(angle_rad);
    return [
      1,    0,   0, 0,
      0,  cos, sin, 0,
      0, -sin, cos, 0,
      0,    0,   0, 1,
    ];
  },

  multiply: (a, b) => {
    const output = new Array(16).fill(0);
    for (let i = 0; i < 4; i++) {
      const ioff = 4 * i;
      for (let j = 0; j < 4; j++) {
        for (let k = 0; k < 4; k++) {
          output[ioff + j] += a[ioff + k] * b[4*k + j];
        }
      }
    }
    return output;
  },

  look_at: function(camera_xyz, target_xyz, up_axis) {
    const z_axis = VEC3.norm(VEC3.subtract(camera_xyz, target_xyz));
    const x_axis = VEC3.norm_cross(up_axis, z_axis);
    const y_axis = VEC3.norm_cross(z_axis, x_axis);
 
    return [
      x_axis[0],     x_axis[1],     x_axis[2],     0,
      y_axis[0],     y_axis[1],     y_axis[2],     0,
      z_axis[0],     z_axis[1],     z_axis[2],     0,
      camera_xyz[0], camera_xyz[1], camera_xyz[2], 1,
    ];
  },

  inverse: function(m) {

    /*
    Invert a 4x4 matrix. Code taken verbatim from:
    https://webgl2fundamentals.org/webgl/lessons/webgl-3d-camera.html
    */

    const m00 = m[0 * 4 + 0];
    const m01 = m[0 * 4 + 1];
    const m02 = m[0 * 4 + 2];
    const m03 = m[0 * 4 + 3];
    const m10 = m[1 * 4 + 0];
    const m11 = m[1 * 4 + 1];
    const m12 = m[1 * 4 + 2];
    const m13 = m[1 * 4 + 3];
    const m20 = m[2 * 4 + 0];
    const m21 = m[2 * 4 + 1];
    const m22 = m[2 * 4 + 2];
    const m23 = m[2 * 4 + 3];
    const m30 = m[3 * 4 + 0];
    const m31 = m[3 * 4 + 1];
    const m32 = m[3 * 4 + 2];
    const m33 = m[3 * 4 + 3];
    const tmp_0  = m22 * m33;
    const tmp_1  = m32 * m23;
    const tmp_2  = m12 * m33;
    const tmp_3  = m32 * m13;
    const tmp_4  = m12 * m23;
    const tmp_5  = m22 * m13;
    const tmp_6  = m02 * m33;
    const tmp_7  = m32 * m03;
    const tmp_8  = m02 * m23;
    const tmp_9  = m22 * m03;
    const tmp_10 = m02 * m13;
    const tmp_11 = m12 * m03;
    const tmp_12 = m20 * m31;
    const tmp_13 = m30 * m21;
    const tmp_14 = m10 * m31;
    const tmp_15 = m30 * m11;
    const tmp_16 = m10 * m21;
    const tmp_17 = m20 * m11;
    const tmp_18 = m00 * m31;
    const tmp_19 = m30 * m01;
    const tmp_20 = m00 * m21;
    const tmp_21 = m20 * m01;
    const tmp_22 = m00 * m11;
    const tmp_23 = m10 * m01;

    const t0 = (tmp_0 * m11 + tmp_3 * m21 + tmp_4 * m31) -
             (tmp_1 * m11 + tmp_2 * m21 + tmp_5 * m31);
    const t1 = (tmp_1 * m01 + tmp_6 * m21 + tmp_9 * m31) -
             (tmp_0 * m01 + tmp_7 * m21 + tmp_8 * m31);
    const t2 = (tmp_2 * m01 + tmp_7 * m11 + tmp_10 * m31) -
             (tmp_3 * m01 + tmp_6 * m11 + tmp_11 * m31);
    const t3 = (tmp_5 * m01 + tmp_8 * m11 + tmp_11 * m21) -
             (tmp_4 * m01 + tmp_9 * m11 + tmp_10 * m21);

    const d = 1.0 / (m00 * t0 + m10 * t1 + m20 * t2 + m30 * t3);

    return [
      d * t0,
      d * t1,
      d * t2,
      d * t3,
      d * ((tmp_1 * m10 + tmp_2 * m20 + tmp_5 * m30) -
           (tmp_0 * m10 + tmp_3 * m20 + tmp_4 * m30)),
      d * ((tmp_0 * m00 + tmp_7 * m20 + tmp_8 * m30) -
           (tmp_1 * m00 + tmp_6 * m20 + tmp_9 * m30)),
      d * ((tmp_3 * m00 + tmp_6 * m10 + tmp_11 * m30) -
           (tmp_2 * m00 + tmp_7 * m10 + tmp_10 * m30)),
      d * ((tmp_4 * m00 + tmp_9 * m10 + tmp_10 * m20) -
           (tmp_5 * m00 + tmp_8 * m10 + tmp_11 * m20)),
      d * ((tmp_12 * m13 + tmp_15 * m23 + tmp_16 * m33) -
           (tmp_13 * m13 + tmp_14 * m23 + tmp_17 * m33)),
      d * ((tmp_13 * m03 + tmp_18 * m23 + tmp_21 * m33) -
           (tmp_12 * m03 + tmp_19 * m23 + tmp_20 * m33)),
      d * ((tmp_14 * m03 + tmp_19 * m13 + tmp_22 * m33) -
           (tmp_15 * m03 + tmp_18 * m13 + tmp_23 * m33)),
      d * ((tmp_17 * m03 + tmp_20 * m13 + tmp_23 * m23) -
           (tmp_16 * m03 + tmp_21 * m13 + tmp_22 * m23)),
      d * ((tmp_14 * m22 + tmp_17 * m32 + tmp_13 * m12) -
           (tmp_16 * m32 + tmp_12 * m12 + tmp_15 * m22)),
      d * ((tmp_20 * m32 + tmp_12 * m02 + tmp_19 * m22) -
           (tmp_18 * m22 + tmp_21 * m32 + tmp_13 * m02)),
      d * ((tmp_18 * m12 + tmp_23 * m32 + tmp_15 * m02) -
           (tmp_22 * m32 + tmp_14 * m02 + tmp_19 * m12)),
      d * ((tmp_22 * m22 + tmp_16 * m02 + tmp_21 * m12) -
           (tmp_20 * m12 + tmp_23 * m22 + tmp_17 * m02)),
    ];
  },

}
