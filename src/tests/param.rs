//! Tests for the `Param` trait implementations.

use crate::{Param, Rot2, Rot3};
use glam::{Vec2, Vec3};

/// Test basic scalar integration for `f32`.
#[test]
fn test_f32_step() {
    // Test with positive derivative
    let x = 1.0f32;
    let dx = 2.0f32;
    let dt = 0.5f32;
    let mut result = x;
    result.step(&dx, dt);
    let expected = x + dx * dt;
    assert!((result - expected).abs() < 1e-6);

    // Test with negative derivative
    let x = 5.0f32;
    let dx = -3.0f32;
    let dt = 2.0f32;
    let mut result = x;
    result.step(&dx, dt);
    let expected = x + dx * dt;
    assert!((result - expected).abs() < 1e-6);

    // Test with zero derivative
    let x = 10.0f32;
    let dx = 0.0f32;
    let dt = 1.0f32;
    let mut result = x;
    result.step(&dx, dt);
    assert!((result - x).abs() < 1e-6);
}

/// Test vector integration for `Vec2`.
#[test]
fn test_vec2_step() {
    let v = Vec2::new(1.0, 2.0);
    let dv = Vec2::new(3.0, -1.0);
    let dt = 0.5f32;
    let mut result = v;
    Param::step(&mut result, &dv, dt);
    let expected = v + dv * dt;
    assert!((result.x - expected.x).abs() < 1e-6);
    assert!((result.y - expected.y).abs() < 1e-6);

    // Test zero time step
    let mut result = v;
    Param::step(&mut result, &dv, 0.0);
    assert!((result.x - v.x).abs() < 1e-6);
    assert!((result.y - v.y).abs() < 1e-6);
}

/// Test vector integration for `Vec3`.
#[test]
fn test_vec3_step() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    let dv = Vec3::new(-1.0, 0.5, 2.0);
    let dt = 0.25f32;
    let mut result = v;
    Param::step(&mut result, &dv, dt);
    let expected = v + dv * dt;
    assert!((result.x - expected.x).abs() < 1e-6);
    assert!((result.y - expected.y).abs() < 1e-6);
    assert!((result.z - expected.z).abs() < 1e-6);
}

/// Test 2D rotation integration.
#[test]
fn test_rot2_step() {
    // Start at angle 0, add angular velocity * dt
    let rot = Rot2::from_angle(0.0);
    let angular_vel = 1.0f32; // 1 rad/s
    let dt = std::f32::consts::FRAC_PI_2; // π/2 seconds

    let mut result = rot;
    result.step(&angular_vel, dt);
    // Should rotate by angular_vel * dt = π/2 radians
    assert!((result.angle() - std::f32::consts::FRAC_PI_2).abs() < 1e-6);

    // Test angle wrapping: rotate by 3π/2 from π/4 should wrap to -π/4 (or 7π/4)
    let rot = Rot2::from_angle(std::f32::consts::FRAC_PI_4);
    let angular_vel = 3.0f32; // 3 rad/s
    let dt = std::f32::consts::FRAC_PI_2; // π/2 seconds => 3π/2 total rotation

    let mut result = rot;
    result.step(&angular_vel, dt);
    // π/4 + 3π/2 = π/4 + 6π/4 = 7π/4 = 2π - π/4
    // Should wrap to equivalent angle in [0, 2π)
    let expected_angle = 7.0 * std::f32::consts::FRAC_PI_4;
    assert!((result.angle() - expected_angle).abs() < 1e-6);
}

/// Test 3D rotation integration.
#[test]
fn test_rot3_step() {
    // Create a rotation from identity, apply angular velocity around z-axis
    let rot = Rot3::default(); // Identity rotation
    let angular_vel = Vec3::new(0.0, 0.0, 1.0); // 1 rad/s around z-axis
    let dt = std::f32::consts::FRAC_PI_2; // π/2 seconds

    let mut result = rot;
    result.step(&angular_vel, dt);

    // After π/2 rotation around z-axis, a vector on x-axis should rotate to y-axis
    let test_vector = Vec3::X;
    let rotated = result.transform(test_vector);
    let expected = Vec3::Y;

    assert!((rotated.x - expected.x).abs() < 1e-6);
    assert!((rotated.y - expected.y).abs() < 1e-6);
    assert!((rotated.z - expected.z).abs() < 1e-6);

    // Test that rotation preserves vector length
    let test_vector = Vec3::new(1.0, 2.0, 3.0);
    let rotated = result.transform(test_vector);
    assert!((rotated.length() - test_vector.length()).abs() < 1e-6);
}
