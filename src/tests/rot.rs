//! Tests for rotation types and utility functions.

use crate::{Rot2, Rot3, angular_to_linear2, angular_to_linear3, torque2, torque3};
use glam::{Vec2, Vec3};
use std::f32::consts::PI;

/// Test Rot2 basic operations.
#[test]
fn test_rot2_basics() {
    // Test from_angle and angle()
    let angle = PI / 3.0;
    let rot = Rot2::from_angle(angle);
    assert!((rot.angle() - angle).abs() < 1e-6);

    // Test angle wrapping
    let rot_wrap = Rot2::from_angle(angle + 2.0 * PI);
    assert!((rot_wrap.angle() - angle).abs() < 1e-6);

    let rot_wrap_negative = Rot2::from_angle(angle - 2.0 * PI);
    assert!((rot_wrap_negative.angle() - angle).abs() < 1e-6);

    // Test angle_degrees
    assert!((rot.angle_degrees() - 60.0).abs() < 1e-6);
}

/// Test Rot2 transformation.
#[test]
fn test_rot2_transform() {
    let rot = Rot2::from_angle(PI / 2.0); // 90 degrees

    // Rotate (1, 0) by 90 degrees -> (0, 1)
    let v = Vec2::new(1.0, 0.0);
    let rotated = rot.transform(v);
    assert!((rotated.x - 0.0).abs() < 1e-6);
    assert!((rotated.y - 1.0).abs() < 1e-6);

    // Rotate (0, 1) by 90 degrees -> (-1, 0)
    let v2 = Vec2::new(0.0, 1.0);
    let rotated2 = rot.transform(v2);
    assert!((rotated2.x + 1.0).abs() < 1e-6);
    assert!((rotated2.y - 0.0).abs() < 1e-6);

    // Test that length is preserved
    let v3 = Vec2::new(3.0, 4.0);
    let rotated3 = rot.transform(v3);
    assert!((rotated3.length() - v3.length()).abs() < 1e-6);
}

/// Test Rot2 chaining and inverse.
#[test]
fn test_rot2_chain_and_inverse() {
    let rot1 = Rot2::from_angle(PI / 4.0); // 45 degrees
    let rot2 = Rot2::from_angle(PI / 4.0); // Another 45 degrees

    // Chain: 45 + 45 = 90 degrees
    let chained = rot1.chain(rot2);
    assert!((chained.angle() - PI / 2.0).abs() < 1e-6);

    // Inverse should undo the rotation
    let inverse = rot1.inverse();
    let identity = rot1.chain(inverse);
    assert!((identity.angle() - 0.0).abs() < 1e-6);

    // Test with transformation
    let v = Vec2::new(1.0, 0.0);
    let rotated = rot1.transform(v);
    let back = inverse.transform(rotated);
    assert!((back.x - v.x).abs() < 1e-6);
    assert!((back.y - v.y).abs() < 1e-6);
}

/// Test Rot2 matrix.
#[test]
fn test_rot2_matrix() {
    let rot = Rot2::from_angle(PI / 3.0);
    let matrix = rot.matrix();

    // Matrix should be orthogonal
    let product = matrix * matrix.transpose();
    assert!((product.x_axis.x - 1.0).abs() < 1e-6);
    assert!((product.x_axis.y - 0.0).abs() < 1e-6);
    assert!((product.y_axis.x - 0.0).abs() < 1e-6);
    assert!((product.y_axis.y - 1.0).abs() < 1e-6);

    // Matrix determinant should be 1 (pure rotation, no scaling)
    assert!((matrix.determinant() - 1.0).abs() < 1e-6);
}

/// Test Rot3 basic operations.
#[test]
fn test_rot3_basics() {
    // Default rotation should be identity
    let identity = Rot3::default();

    // Identity should not change vectors
    let v = Vec3::new(1.0, 2.0, 3.0);
    let transformed = identity.transform(v);
    assert!((transformed.x - v.x).abs() < 1e-6);
    assert!((transformed.y - v.y).abs() < 1e-6);
    assert!((transformed.z - v.z).abs() < 1e-6);

    // Test from_scaled_axis
    let axis = Vec3::new(0.0, 0.0, 1.0); // Z-axis
    let angle = PI / 2.0;
    let rot = Rot3::from_scaled_axis(axis * angle);

    // Rotate (1, 0, 0) by 90 degrees around Z -> (0, 1, 0)
    let v = Vec3::new(1.0, 0.0, 0.0);
    let rotated = rot.transform(v);
    assert!((rotated.x - 0.0).abs() < 1e-6);
    assert!((rotated.y - 1.0).abs() < 1e-6);
    assert!((rotated.z - 0.0).abs() < 1e-6);
}

/// Test Rot3 chaining and inverse.
#[test]
fn test_rot3_chain_and_inverse() {
    // Rotation around X-axis by 90 degrees
    let rot_x = Rot3::from_scaled_axis(Vec3::new(1.0, 0.0, 0.0) * PI / 2.0);

    // Rotation around Y-axis by 90 degrees
    let rot_y = Rot3::from_scaled_axis(Vec3::new(0.0, 1.0, 0.0) * PI / 2.0);

    // Chain rotations: first X, then Y
    let chained = rot_x.chain(rot_y);

    // Test that chaining is not commutative
    let chained_reverse = rot_y.chain(rot_x);

    let v = Vec3::new(1.0, 0.0, 0.0);
    let result1 = chained.transform(v);
    let result2 = chained_reverse.transform(v);

    // These should be different (rotations don't commute)
    assert!((result1 - result2).length() > 0.1);

    // Inverse should undo the rotation
    let inverse = rot_x.inverse();
    let identity_like = rot_x.chain(inverse);

    let v_test = Vec3::new(1.0, 2.0, 3.0);
    let back = identity_like.transform(v_test);
    assert!((back - v_test).length() < 1e-6);
}

/// Test Rot3 matrix properties.
#[test]
fn test_rot3_matrix() {
    let axis = Vec3::new(1.0, 2.0, 3.0).normalize();
    let angle = PI / 4.0;
    let rot = Rot3::from_scaled_axis(axis * angle);
    let matrix = rot.matrix();

    // Rotation matrix should be orthogonal: M * M^T = I
    let product = matrix * matrix.transpose();

    // Check diagonal elements are close to 1, off-diagonal close to 0
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((product.col(i)[j] - expected).abs() < 1e-6);
        }
    }

    // Determinant should be 1 (proper rotation, no reflection)
    assert!((matrix.determinant() - 1.0).abs() < 1e-6);
}

/// Test torque2 function.
#[test]
fn test_torque2() {
    // Position vector (lever arm)
    let pos = Vec2::new(2.0, 0.0);

    // Force perpendicular to lever arm (max torque)
    let force = Vec2::new(0.0, 3.0);

    // τ = r × F = r_x * F_y - r_y * F_x
    // = 2.0 * 3.0 - 0.0 * 0.0 = 6.0
    let torque = torque2(pos, force);
    assert!((torque - 6.0).abs() < 1e-6);

    // Force parallel to lever arm (zero torque)
    let force_parallel = Vec2::new(4.0, 0.0);
    let torque_parallel = torque2(pos, force_parallel);
    assert!((torque_parallel - 0.0).abs() < 1e-6);

    // Test with negative torque (clockwise)
    let force_negative = Vec2::new(0.0, -3.0);
    let torque_negative = torque2(pos, force_negative);
    assert!((torque_negative + 6.0).abs() < 1e-6);
}

/// Test torque3 function.
#[test]
fn test_torque3() {
    // Position vector (lever arm)
    let pos = Vec3::new(2.0, 0.0, 0.0);

    // Force perpendicular to lever arm (max torque)
    let force = Vec3::new(0.0, 3.0, 0.0);

    // τ = r × F = (0, 0, r_x*F_y - r_y*F_x) = (0, 0, 6.0)
    let torque = torque3(pos, force);
    assert!((torque.x - 0.0).abs() < 1e-6);
    assert!((torque.y - 0.0).abs() < 1e-6);
    assert!((torque.z - 6.0).abs() < 1e-6);

    // Force parallel to lever arm (zero torque)
    let force_parallel = Vec3::new(4.0, 0.0, 0.0);
    let torque_parallel = torque3(pos, force_parallel);
    assert!(torque_parallel.length() < 1e-6);

    // Test cross product properties
    let pos2 = Vec3::new(1.0, 2.0, 3.0);
    let force2 = Vec3::new(4.0, 5.0, 6.0);
    let torque2 = torque3(pos2, force2);

    // Torque should be perpendicular to both position and force
    assert!((torque2.dot(pos2)).abs() < 1e-6);
    assert!((torque2.dot(force2)).abs() < 1e-6);
}

/// Test angular_to_linear2 function.
#[test]
fn test_angular_to_linear2() {
    // Angular velocity (positive = counter-clockwise)
    let angular = 2.0;

    // Position vector from rotation center
    let pos = Vec2::new(3.0, 0.0);

    // v = ω × r = ω * (-r_y, r_x) = 2.0 * (0, 3) = (0, 6)
    let linear = angular_to_linear2(angular, pos);
    assert!((linear.x - 0.0).abs() < 1e-6);
    assert!((linear.y - 6.0).abs() < 1e-6);

    // Test with position at angle
    let pos2 = Vec2::new(1.0, 1.0);
    let linear2 = angular_to_linear2(angular, pos2);

    // Velocity should be perpendicular to position
    assert!((linear2.dot(pos2)).abs() < 1e-6);

    // Magnitude should be ω * |r|
    let expected_magnitude = angular * pos2.length();
    assert!((linear2.length() - expected_magnitude).abs() < 1e-6);
}

/// Test angular_to_linear3 function.
#[test]
fn test_angular_to_linear3() {
    // Angular velocity vector (around Z-axis)
    let angular = Vec3::new(0.0, 0.0, 2.0);

    // Position vector from rotation center
    let pos = Vec3::new(3.0, 0.0, 0.0);

    // v = ω × r = (0, 0, 2) × (3, 0, 0) = (0, 6, 0)
    let linear = angular_to_linear3(angular, pos);
    assert!((linear.x - 0.0).abs() < 1e-6);
    assert!((linear.y - 6.0).abs() < 1e-6);
    assert!((linear.z - 0.0).abs() < 1e-6);

    // Velocity should be perpendicular to both angular velocity and position
    assert!((linear.dot(angular)).abs() < 1e-6);
    assert!((linear.dot(pos)).abs() < 1e-6);

    // Test magnitude: |v| = |ω| * |r| * sin(θ)
    // For perpendicular vectors, sin(θ) = 1
    let angular_mag = angular.length();
    let pos_mag = pos.length();
    let expected_magnitude = angular_mag * pos_mag;
    assert!((linear.length() - expected_magnitude).abs() < 1e-6);
}
