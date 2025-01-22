use crate::types::{ray_sphere_intersection, Ray, Sphere};
use vecmath::vec3_sub;

#[derive(Clone)]
pub struct BoundingBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

pub enum BVHNode {
    Leaf {
        sphere: Sphere,
    },
    Internal {
        left: Box<BVHNode>,
        right: Box<BVHNode>,
        bbox: BoundingBox,
    },
}

pub fn build_bvh(spheres: Vec<Sphere>) -> BVHNode {
    if spheres.len() == 1 {
        return BVHNode::Leaf {
            sphere: spheres[0].clone(),
        };
    }

    let mut min = [f32::INFINITY, f32::INFINITY, f32::INFINITY];
    let mut max = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];

    for sphere in &spheres {
        for i in 0..3 {
            min[i] = f32::min(min[i], sphere.xyz[i] - sphere.r);
            max[i] = f32::max(max[i], sphere.xyz[i] + sphere.r);
        }
    }
    // identify from all the spheres the space they all span,
    // lower most x coord, lower most y coord, lower most z coord, and higher
    // potential to precompute this w a range tree or sum shit

    let axis = {
        let size = vec3_sub(max, min);
        if size[0] > size[1] && size[0] > size[2] {
            0
        } else if size[1] > size[2] {
            1
        } else {
            2
        }
    };

    //axis to split on

    let mut spheres = spheres;
    spheres.sort_by(|a, b| a.xyz[axis].partial_cmp(&b.xyz[axis]).unwrap());

    let mid = spheres.len() / 2;
    let left_spheres = spheres[..mid].to_vec();
    let right_spheres = spheres[mid..].to_vec();

    let left = build_bvh(left_spheres);
    let right = build_bvh(right_spheres);

    BVHNode::Internal {
        left: Box::new(left),
        right: Box::new(right),
        bbox: BoundingBox { min, max },
    }
}

pub fn bvh_intersection<'a>(ray: &Ray, node: &'a BVHNode) -> Option<(f32, [f32; 3], &'a Sphere)> {
    match node {
        BVHNode::Leaf { sphere } => {
            if let Some((t, intersection_point)) = ray_sphere_intersection(ray, sphere) {
                return Some((t, intersection_point, sphere));
            }
        }
        BVHNode::Internal { left, right, bbox } => {
            if ray_bbox_intersection(ray, bbox) {
                let left_hit = bvh_intersection(ray, left);
                let right_hit = bvh_intersection(ray, right);

                return match (left_hit, right_hit) {
                    (Some(left), Some(right)) => {
                        if left.0 < right.0 {
                            Some(left)
                        } else {
                            Some(right)
                        }
                    }
                    (Some(left), None) => Some(left),
                    (None, Some(right)) => Some(right),
                    (None, None) => None,
                };
            }
        }
    }
    None
}

fn ray_bbox_intersection(ray: &Ray, bbox: &BoundingBox) -> bool {
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;

    for i in 0..3 {
        let inv_d = 1.0 / ray.direction[i];
        let t0 = (bbox.min[i] - ray.origin[i]) * inv_d;
        let t1 = (bbox.max[i] - ray.origin[i]) * inv_d;

        let (t0, t1) = if inv_d < 0.0 { (t1, t0) } else { (t0, t1) };
        t_min = t_min.max(t0);
        t_max = t_max.min(t1);

        if t_max < t_min {
            return false;
        }
    }

    true
}
