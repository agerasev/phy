use crate::Param;
use core::f32::consts::PI;
use glam::{Mat2, Mat3, Quat, Vec2, Vec3};

/// 2D Rotation.
#[derive(Clone, Copy, Default, Debug)]
pub struct Rot2(f32);

/// 3D Rotation.
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

impl Rot2 {
    /// From angle in radians
    pub fn from_angle(angle: f32) -> Self {
        Self(angle % (2.0 * PI))
    }

    /// Angle in radians `0.0..(2.0 * PI)`
    pub fn angle(self) -> f32 {
        self.0
    }
    /// Angle in degrees `0.0..360.0`
    pub fn angle_degrees(self) -> f32 {
        (180.0 / PI) * self.angle()
    }
    pub fn matrix(self) -> Mat2 {
        Mat2::from_angle(self.0)
    }

    pub fn transform(&self, v: Vec2) -> Vec2 {
        self.matrix().mul_vec2(v)
    }
    pub fn chain(self, other: Self) -> Self {
        Self((self.0 + other.0) % (2.0 * PI))
    }
    pub fn inverse(self) -> Self {
        Self(-self.0)
    }
}

impl Rot3 {
    pub fn from_scaled_axis(v: Vec3) -> Self {
        Self(Quat::from_scaled_axis(v))
    }

    pub fn matrix(self) -> Mat3 {
        Mat3::from_quat(self.0)
    }

    pub fn transform(self, v: Vec3) -> Vec3 {
        self.0.mul_vec3(v)
    }
    pub fn chain(self, other: Self) -> Self {
        Self(other.0.mul_quat(self.0).normalize())
    }
    pub fn inverse(self) -> Self {
        Self(self.0.inverse())
    }
}

impl Param for Rot2 {
    /// Angular speed
    type Deriv = f32;
    fn step(self, dp: f32, dt: f32) -> Self {
        self.chain(Rot2::from_angle(dp * dt))
    }
}
impl Param for Rot3 {
    /// Direction is an axis of rotation.
    /// Length is angular speed around this axis.
    type Deriv = Vec3;
    fn step(self, dp: Vec3, dt: f32) -> Self {
        self.chain(Rot3::from_scaled_axis(dp * dt))
    }
}

pub fn torque2(pos: Vec2, vec: Vec2) -> f32 {
    pos.perp_dot(vec)
}
pub fn torque3(pos: Vec3, vec: Vec3) -> Vec3 {
    pos.cross(vec)
}

pub fn angular_to_linear2(angular: f32, pos: Vec2) -> Vec2 {
    angular * pos.perp()
}
pub fn angular_to_linear3(angular: Vec3, pos: Vec3) -> Vec3 {
    angular.cross(pos)
}
