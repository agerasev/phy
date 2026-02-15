use crate::Param;
use core::f32::consts::PI;
use glam::{Mat2, Mat3, Quat, Vec2, Vec3};

/// 2D rotation represented by an angle in radians.
///
/// The angle is stored in radians and is kept within the range [0, 2π)
/// through modulo arithmetic. This ensures consistent representation
/// while preserving the mathematical properties of rotation.
#[derive(Clone, Copy, Default, Debug)]
pub struct Rot2(f32);

/// 3D rotation represented by a unit quaternion.
///
/// Uses `glam::Quat` internally to represent 3D rotations.
/// The quaternion is always normalized to maintain unit length.
#[derive(Clone, Copy, Debug)]
pub struct Rot3(Quat);

impl From<f32> for Rot2 {
    fn from(value: f32) -> Self {
        Self(value)
    }
}
impl From<Rot2> for f32 {
    fn from(value: Rot2) -> Self {
        value.0
    }
}

impl From<Quat> for Rot3 {
    fn from(value: Quat) -> Self {
        Self(value)
    }
}
impl From<Rot3> for Quat {
    fn from(value: Rot3) -> Self {
        value.0
    }
}

impl Default for Rot3 {
    fn default() -> Self {
        Self(Quat::IDENTITY)
    }
}

/// Wrap an angle to the range [0, 2π) using Euclidean remainder.
#[cfg(feature = "std")]
fn wrap_angle(angle: f32) -> f32 {
    angle.rem_euclid(2.0 * PI)
}

/// Wrap an angle to the range [0, 2π) using Euclidean remainder.
#[cfg(not(feature = "std"))]
fn wrap_angle(angle: f32) -> f32 {
    libm::remainderf(angle, 2.0 * PI)
}

impl Rot2 {
    /// Create a 2D rotation from an angle in radians.
    ///
    /// The angle is wrapped into the range [0, 2π) using Euclidean
    /// remainder for improved numerical stability.
    ///
    /// # Examples
    /// ```
    /// use phy::Rot2;
    /// use core::f32::consts::PI;
    ///
    /// let rot = Rot2::from_angle(3.0 * PI);
    /// assert!((rot.angle() - PI).abs() < 1e-6);
    /// ```
    pub fn from_angle(angle: f32) -> Self {
        Self(wrap_angle(angle))
    }

    /// Get the angle in radians, in the range [0, 2π)
    pub fn angle(self) -> f32 {
        self.0
    }

    /// Get the angle in degrees, in the range [0, 360)
    pub fn angle_degrees(self) -> f32 {
        (180.0 / PI) * self.angle()
    }

    /// Get the 2D rotation matrix.
    ///
    /// Returns a 2x2 rotation matrix that can be used to transform vectors.
    pub fn matrix(self) -> Mat2 {
        Mat2::from_angle(self.0)
    }

    /// Transform a 2D vector by this rotation.
    ///
    /// Applies the rotation to the given vector, returning the rotated vector.
    pub fn transform(&self, v: Vec2) -> Vec2 {
        self.matrix().mul_vec2(v)
    }

    /// Chain this rotation with another rotation.
    ///
    /// Returns a new rotation that represents applying `self` then `other`.
    /// The resulting angle is wrapped to [0, 2π).
    pub fn chain(self, other: Self) -> Self {
        Self(wrap_angle(self.0 + other.0))
    }

    /// Get the inverse rotation.
    ///
    /// Returns a rotation that undoes this rotation.
    pub fn inverse(self) -> Self {
        Self(-self.0)
    }
}

impl Rot3 {
    /// Create a 3D rotation from an axis-angle representation.
    ///
    /// # Arguments
    /// * `v` - A vector where the direction represents the rotation axis
    ///   and the magnitude represents the rotation angle in radians.
    ///
    /// # Examples
    /// ```
    /// use phy::Rot3;
    /// use glam::Vec3;
    ///
    /// // 90-degree rotation around the z-axis
    /// let rot = Rot3::from_scaled_axis(Vec3::Z * std::f32::consts::FRAC_PI_2);
    /// ```
    pub fn from_scaled_axis(v: Vec3) -> Self {
        Self(Quat::from_scaled_axis(v))
    }

    /// Get the 3D rotation matrix.
    ///
    /// Returns a 3x3 rotation matrix that can be used to transform vectors.
    pub fn matrix(self) -> Mat3 {
        Mat3::from_quat(self.0)
    }

    /// Transform a 3D vector by this rotation.
    ///
    /// Applies the rotation to the given vector, returning the rotated vector.
    pub fn transform(self, v: Vec3) -> Vec3 {
        self.0.mul_vec3(v)
    }

    /// Chain this rotation with another rotation.
    ///
    /// Returns a new rotation that represents applying `self` then `other`.
    /// The resulting quaternion is normalized to maintain unit length.
    ///
    /// # Arguments
    /// * `other` - The rotation to apply after this one.
    pub fn chain(self, other: Self) -> Self {
        Self(other.0.mul_quat(self.0).normalize())
    }

    /// Get the inverse rotation.
    ///
    /// Returns a rotation that undoes this rotation.
    pub fn inverse(self) -> Self {
        Self(self.0.inverse())
    }
}

impl Param for Rot2 {
    /// Angular speed in radians per unit time.
    type Deriv = f32;

    /// Advance the rotation by integrating angular velocity over time.
    ///
    /// Computes: `angle_{n+1} = angle_n + ω * dt` (modulo 2π)
    /// where ω is the angular velocity (`dp`).
    fn step(&mut self, dp: &f32, dt: f32) {
        *self = self.chain(Rot2::from_angle(dp * dt));
    }
}

impl Param for Rot3 {
    /// Angular velocity vector.
    ///
    /// The direction is the axis of rotation, and the magnitude is the
    /// angular speed in radians per unit time.
    type Deriv = Vec3;

    /// Advance the rotation by integrating angular velocity over time.
    ///
    /// Computes a rotation increment from the axis-angle representation
    /// of `dp * dt` and chains it with the current rotation.
    fn step(&mut self, dp: &Vec3, dt: f32) {
        *self = self.chain(Rot3::from_scaled_axis(dp * dt));
    }
}

/// Compute the moment of force (torque) in 2D.
///
/// In 2D, torque is a scalar representing the magnitude of rotational force.
///
/// # Arguments
/// * `pos` - Position vector where force is applied, relative to rotation axis.
/// * `force` - Force vector applied at that position.
///
/// # Returns
/// The torque value (positive for counter-clockwise, negative for clockwise).
///
/// # Formula
/// `τ = r × F = r_x * F_y - r_y * F_x`
pub fn torque2(pos: Vec2, force: Vec2) -> f32 {
    pos.perp_dot(force)
}

/// Compute the moment of force (torque) in 3D.
///
/// In 3D, torque is a vector where direction is the rotation axis
/// and magnitude is the torque magnitude.
///
/// # Arguments
/// * `pos` - Position vector where force is applied, relative to rotation point.
/// * `force` - Force vector applied at that position.
///
/// # Returns
/// The torque vector: `τ = r × F`
pub fn torque3(pos: Vec3, force: Vec3) -> Vec3 {
    pos.cross(force)
}

/// Compute linear velocity at a point due to angular velocity in 2D.
///
/// For a rigid body rotating with angular velocity `ω` about the origin,
/// the linear velocity at point `r` is `v = ω × r` (perpendicular to `r`).
///
/// # Arguments
/// * `angular` - Angular velocity (scalar, positive for counter-clockwise).
/// * `pos` - Position vector relative to rotation center.
///
/// # Returns
/// Linear velocity vector at the given position.
///
/// # Formula
/// `v = ω * (-r_y, r_x)`
pub fn angular_to_linear2(angular: f32, pos: Vec2) -> Vec2 {
    angular * pos.perp()
}

/// Compute linear velocity at a point due to angular velocity in 3D.
///
/// For a rigid body rotating with angular velocity vector `ω` about a point,
/// the linear velocity at point `r` is `v = ω × r`.
///
/// # Arguments
/// * `angular` - Angular velocity vector (direction is rotation axis,
///   magnitude is angular speed).
/// * `pos` - Position vector relative to rotation center.
///
/// # Returns
/// Linear velocity vector at the given position.
pub fn angular_to_linear3(angular: Vec3, pos: Vec3) -> Vec3 {
    angular.cross(pos)
}
