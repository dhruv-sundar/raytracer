use rand_distr::{Distribution, Normal};
use vecmath::{vec3_add, vec3_dot, vec3_len, vec3_scale, vec3_sub};

pub struct Ray {
    pub origin: [f32; 3],
    pub direction: [f32; 3],
}

#[derive(Clone)]
pub struct Sphere {
    pub xyz: [f32; 3],
    pub r: f32,
    pub rgb: [f32; 3],
    pub shininess: [f32; 3],
    pub roughness: Option<Normal<f32>>,
}

pub struct Plane {
    pub normal: [f32; 3], // Normal vector (A, B, C)
    pub d: f32,           // D in the plane equation Ax + By + Cz + D = 0
    pub color: [f32; 3],  // Color of the plane
}

#[derive(Clone)]
pub struct Triangle {
    pub v0: [f32; 3],             // First vertex of the triangle
    pub v1: [f32; 3],             // Second vertex of the triangle
    pub v2: [f32; 3],             // Third vertex of the triangle
    pub color: [f32; 3],          // Color of the triangle
    pub normal: Option<[f32; 3]>, // Optional precomputed normal
}

pub fn ray_sphere_intersection(ray: &Ray, sphere: &Sphere) -> Option<(f32, [f32; 3])> {
    let sphere_center = sphere.xyz;

    let oc = vec3_sub(sphere_center, ray.origin);
    let oc_mag = vec3_len(oc);

    let inside = (oc_mag * oc_mag) < (sphere.r * sphere.r);

    let rd_len = vec3_len(ray.direction);
    let tc = vec3_dot(oc, ray.direction) / rd_len;

    if !inside && tc < 0.0 {
        return None;
    }

    let tcrd = vec3_scale(ray.direction, tc);
    let ro_tcrd = vec3_add(ray.origin, tcrd);
    let d = vec3_len(vec3_sub(ro_tcrd, sphere_center));
    let d_squared = d * d;

    let r_squared = sphere.r * sphere.r;
    if !inside && r_squared < d_squared {
        return None;
    }

    // Step 6: Compute `t_offset`
    let t_offset = (r_squared - d_squared).sqrt() / rd_len;

    let t1 = tc + t_offset;
    let t2 = tc - t_offset;

    let t = if inside { t1 } else { t2 };

    // Compute intersection point: p = r_o + t * r_d
    let intersection_point = vec3_add(vec3_scale(ray.direction, t), ray.origin);

    Some((t, intersection_point))
}
