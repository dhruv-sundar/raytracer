extern crate vecmath;
mod bvh;
mod types;

extern crate rayon;
use rayon::prelude::*;

use crate::types::{Ray, Sphere};
use rand::Rng;
use rand::{rngs::ThreadRng, thread_rng};
use rand_distr::{Distribution, Normal};

use image::RgbaImage;
use std::env;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use bvh::{build_bvh, bvh_intersection, BVHNode};

use vecmath::{
    vec3_add, vec3_cross, vec3_dot, vec3_len, vec3_mul, vec3_neg, vec3_normalized, vec3_scale,
    vec3_sub,
};

struct Sun {
    xyz: [f32; 3],
    rgb: [f32; 3],
    is_bulb: bool,
}

struct RenderState {
    height: u32,
    width: u32,
    filename: String,
    color: [f32; 3],
    suns: Vec<Sun>,
    spheres: Vec<Sphere>,
    bvh: Option<BVHNode>,
    forward: [f32; 3],
    up: [f32; 3],
    right: [f32; 3],
    eye: [f32; 3],
    exposure: Option<f32>,
    rand: Option<f32>,
    bounces: u8,
    shininess: [f32; 3],
    ior: f32,
    dof: Option<[f32; 2]>,
    aa: u8,
}
impl RenderState {
    // Function to initialize with default values
    fn new() -> Self {
        RenderState {
            height: 0,
            width: 0,
            filename: String::new(),
            color: [1.0, 1.0, 1.0], // Fixed-size array
            suns: Vec::new(),
            spheres: Vec::new(),
            bvh: None,
            forward: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
            eye: [0.0, 0.0, 0.0],
            exposure: None,
            rand: None,
            bounces: 4,
            shininess: [0.0, 0.0, 0.0],
            ior: 1.458,
            dof: None,
            aa: 1,
        }
    }
}

fn sx_sy(x: f32, y: f32, state: &RenderState) -> (f32, f32) {
    let w: f32 = state.width as f32;
    let h: f32 = state.height as f32;
    let denom: f32 = f32::max(w, h);

    ((2.0 * x - w) / denom, (h - 2.0 * y) / denom)
}

fn ray_from_s(sx: f32, sy: f32, state: &RenderState) -> Ray {
    let scaled_right = vec3_scale(state.right, sx);
    let scaled_up = vec3_scale(state.up, sy);

    let direction_unnormalized = vec3_add(vec3_add(state.forward, scaled_right), scaled_up);

    let direction = vec3_normalized(direction_unnormalized);

    Ray {
        origin: state.eye,
        direction,
    }
}

fn perturb_normal(normal: [f32; 3], sphere: &Sphere) -> [f32; 3] {
    if let Some(rand) = &sphere.roughness {
        let delta_x = rand.sample(&mut rand::thread_rng());
        let delta_y = rand.sample(&mut rand::thread_rng());
        let delta_z = rand.sample(&mut rand::thread_rng());

        return vec3_normalized([
            normal[0] + delta_x,
            normal[1] + delta_y,
            normal[2] + delta_z,
        ]);
    }
    normal
}

fn apply_dof(state: &RenderState, ray: &Ray, rng: &mut ThreadRng, focus: f32, lens: f32) {
    //o_old + t * d_old = o_new + t * d_new
    let random_angle = rng.gen::<f32>() * 2.0 * 3.14;
    let random_radius = rng.gen::<f32>().sqrt() * lens;
    //state.eye[0] + random_radius * random_angle.cos() + state.up[0] * random_radius * random_angle.sin(),
}

fn apply_exposure(l: f32, state: &RenderState) -> f32 {
    if let Some(exposure) = state.exposure {
        return 1.0 - (-exposure * l).exp();
    }
    l
}

fn linear_to_srgb(l: f32) -> f32 {
    let clamped = num::clamp(l, 0.0, 1.0);
    if clamped <= 0.0031308 {
        12.92 * clamped
    } else {
        1.055 * clamped.powf(1.0 / 2.4) - 0.055
    }
}

fn lambert(
    normal: [f32; 3],
    ray_dir: [f32; 3],
    light_dir: [f32; 3],
    light_color: [f32; 3],
    object_color: [f32; 3],
) -> [f32; 3] {
    //dotting normal with ray direction, not light direction
    //object color * light color + dot(normal, ray dir)
    let dot = vec3_dot(normal, ray_dir);
    let corrected_normal = if dot > 0.0 { vec3_neg(normal) } else { normal };

    let lambertian = f32::max(0.0, vec3_dot(corrected_normal, light_dir));

    let mut result = vec3_scale(object_color, lambertian);
    result = vec3_mul(result, light_color);

    result
}

fn compute_reflection(state: &RenderState, reflected_ray: &Ray, bounces: u8) -> [f32; 3] {
    if let Some(bvh) = &state.bvh {
        if let Some((t, intersection_point, sphere)) = bvh_intersection(reflected_ray, bvh) {
            let illumination =
                compute_illumination(state, sphere, intersection_point, reflected_ray.direction);

            if bounces == 0 {
                //base case, 0 shininess, 100 illumination
                //further calls will be shininess * recurse() + (1 - shininess)*illumination
                // so then ray that hits A and bounces once on B will have shininess * illuminate(B) + (1 - shininess) * illuminate(A)
                // computeReflection will hit B returns illuminate(B), in computeIlluminate, will be scaled accordingly by shininess, and illuminate(A) also
                // computed accordingly
                return illumination;
            }

            let mut normal = vec3_normalized(vec3_sub(intersection_point, sphere.xyz));
            normal = perturb_normal(normal, sphere);

            let reflected_dir = vec3_normalized(vec3_sub(
                reflected_ray.direction,
                vec3_scale(normal, 2.0 * vec3_dot(reflected_ray.direction, normal)),
            ));

            let further_reflections = compute_reflection(
                state,
                &Ray {
                    origin: vec3_add(intersection_point, vec3_scale(reflected_dir, 1e-3)),
                    direction: reflected_dir,
                },
                bounces - 1,
            );

            let reflection_val = vec3_mul(further_reflections, sphere.shininess);
            let shinied_illumination =
                vec3_mul(illumination, vec3_sub([1.0, 1.0, 1.0], sphere.shininess));

            return vec3_add(shinied_illumination, reflection_val);
        }
    }

    [0.0, 0.0, 0.0]
}

fn compute_illumination(
    state: &RenderState,
    sphere: &Sphere,
    intersection_point: [f32; 3],
    ray_direction: [f32; 3],
) -> [f32; 3] {
    let mut total_color = [0.0, 0.0, 0.0];
    let epsilon = 1e-4;

    let sphere_color = sphere.rgb;
    let sphere_center = sphere.xyz;

    for sun in &state.suns {
        let mut light_dir = vec3_normalized(sun.xyz);
        if sun.is_bulb {
            light_dir = vec3_normalized(vec3_sub(sun.xyz, intersection_point));
        }

        let shadow_ray_origin = vec3_add(intersection_point, vec3_scale(light_dir, epsilon));
        let shadow_ray = Ray {
            origin: shadow_ray_origin,
            direction: light_dir,
        };

        // Check shadows
        let mut in_shadow = false;
        if let Some(bvh) = &state.bvh {
            if let Some((t, _, _)) = bvh_intersection(&shadow_ray, bvh) {
                let light_distance = vec3_len(vec3_sub(sun.xyz, shadow_ray_origin));
                if t > 0.0 && t < light_distance {
                    in_shadow = true;
                }
            }
        }

        if !in_shadow {
            let mut normal = vec3_normalized(vec3_sub(intersection_point, sphere_center));
            normal = perturb_normal(normal, sphere);

            let mut light_contribution = sun.rgb;

            if sun.is_bulb {
                let distance = vec3_len(vec3_sub(sun.xyz, intersection_point));
                light_contribution = vec3_scale(light_contribution, 1.0 / (distance * distance));
            }

            let diffuse = lambert(
                normal,
                ray_direction,
                light_dir,
                light_contribution,
                sphere_color,
            );

            total_color = vec3_add(total_color, diffuse);
        }
    }

    total_color
}

fn illuminate(
    state: &RenderState,
    sphere: &Sphere,
    intersection_point: [f32; 3],
    ray_direction: [f32; 3],
    bounces: u8,
) -> [f32; 3] {
    let sphere_color = sphere.rgb;
    let sphere_center = sphere.xyz;
    let shininess = sphere.shininess;

    let mut total_color = [0.0, 0.0, 0.0];

    let diffuse = compute_illumination(state, sphere, intersection_point, ray_direction);

    let mut normal = vec3_normalized(vec3_sub(intersection_point, sphere_center));
    normal = perturb_normal(normal, sphere);

    let reflected_dir = vec3_normalized(vec3_sub(
        ray_direction,
        vec3_scale(normal, 2.0 * vec3_dot(ray_direction, normal)),
    ));

    let reflected_ray = Ray {
        origin: vec3_add(intersection_point, vec3_scale(reflected_dir, 1e-3)),
        direction: reflected_dir,
    };

    let reflected_color = compute_reflection(state, &reflected_ray, bounces - 1);

    // Add the reflected contribution weighted by shininess
    total_color = vec3_add(
        vec3_mul(reflected_color, shininess),
        vec3_mul(diffuse, vec3_sub([1.0, 1.0, 1.0], shininess)),
    );

    total_color
}

fn compute_rays(state: &RenderState) -> Option<RgbaImage> {
    if let Some(bvh) = &state.bvh {
        let mut img = RgbaImage::new(state.width, state.height);
        let (width, height) = (state.width, state.height);

        let pixels: Vec<(u32, u32, [f32; 3])> = (0..height)
            .into_par_iter()
            .flat_map(|y| {
                (0..width).into_par_iter().map(move |x| {
                    let (sx, sy) = sx_sy(x as f32, y as f32, state);
                    let ray = ray_from_s(sx, sy, state);
                    if let Some((_t, point, sphere)) = bvh_intersection(&ray, bvh) {
                        let color =
                            illuminate(state, sphere, point, ray.direction, state.bounces - 1);
                        let converted = [
                            linear_to_srgb(apply_exposure(color[0], state)),
                            linear_to_srgb(apply_exposure(color[1], state)),
                            linear_to_srgb(apply_exposure(color[2], state)),
                        ];
                        (x, y, converted)
                    } else {
                        (x, y, [0.0, 0.0, 0.0])
                    }
                })
            })
            .collect();

        for (x, y, color) in pixels {
            let pixel = img.get_pixel_mut(x, y);
            *pixel = image::Rgba([
                (color[0] * 255.0) as u8,
                (color[1] * 255.0) as u8,
                (color[2] * 255.0) as u8,
                255,
            ]);
        }

        Some(img)
    } else {
        None
    }
}

fn process_line(line: &str, state: &mut RenderState) {
    let tokens: Vec<&str> = line.split_whitespace().collect();

    match tokens.get(0) {
        Some(&"png") => {
            if tokens.len() >= 4 {
                let width: u32 = tokens[1].parse().unwrap();
                let height: u32 = tokens[2].parse().unwrap();
                let filename = tokens[3];
                println!(
                    "PNG: width = {}, height = {}, filename = {}",
                    width, height, filename
                );

                state.filename = filename.to_string();
                state.height = height;
                state.width = width;
            } else {
                eprintln!("Invalid PNG command format");
            }
        }
        Some(&"color") => {
            if tokens.len() >= 4 {
                let r: f32 = tokens[1].parse().unwrap();
                let g: f32 = tokens[2].parse().unwrap();
                let b: f32 = tokens[3].parse().unwrap();

                state.color = [r, g, b];
            } else {
                eprintln!("Invalid RGB command format");
            }
        }
        Some(&"sun") => {
            if tokens.len() >= 4 {
                let x: f32 = tokens[1].parse().unwrap();
                let y: f32 = tokens[2].parse().unwrap();
                let z: f32 = tokens[3].parse().unwrap();
                println!("Sun: x = {}, y = {}, z = {}", x, y, z);

                let sun = Sun {
                    xyz: [x, y, z],
                    rgb: state.color.clone(),
                    is_bulb: false,
                };

                state.suns.push(sun);
            } else {
                eprintln!("Invalid Sun command format");
            }
        }
        Some(&"bulb") => {
            if tokens.len() >= 4 {
                let x: f32 = tokens[1].parse().unwrap();
                let y: f32 = tokens[2].parse().unwrap();
                let z: f32 = tokens[3].parse().unwrap();
                println!("Bulb: x = {}, y = {}, z = {}", x, y, z);

                let bulb = Sun {
                    xyz: [x, y, z],
                    rgb: state.color.clone(),
                    is_bulb: true,
                };

                state.suns.push(bulb);
            } else {
                eprintln!("Invalid Bulb command format");
            }
        }
        Some(&"sphere") => {
            if tokens.len() >= 5 {
                let x: f32 = tokens[1].parse().unwrap();
                let y: f32 = tokens[2].parse().unwrap();
                let z: f32 = tokens[3].parse().unwrap();
                let r: f32 = tokens[4].parse().unwrap();

                let normal_dist = if let Some(sigma) = state.rand {
                    Some(Normal::new(0.0, sigma).expect("Failed to create distribution"))
                } else {
                    None
                };

                let sphere = Sphere {
                    xyz: [x, y, z],
                    r: r,
                    rgb: state.color.clone(),
                    shininess: state.shininess.clone(),
                    roughness: normal_dist,
                };
                state.spheres.push(sphere);
            } else {
                eprintln!("Invalid Sun command format");
            }
        }
        Some(&"expose") => {
            if tokens.len() >= 2 {
                let v: f32 = tokens[1].parse().unwrap();

                state.exposure = Some(v);

                println!("exposure set to {}", v);
            } else {
                eprintln!("Invalid exposure format");
            }
        }
        Some(&"eye") => {
            if tokens.len() >= 4 {
                let x: f32 = tokens[1].parse().unwrap();
                let y: f32 = tokens[2].parse().unwrap();
                let z: f32 = tokens[3].parse().unwrap();

                state.eye = [x, y, z];

                println!("eye set to {} {} {}", x, y, z);
            } else {
                eprintln!("Invalid eye format");
            }
        }
        Some(&"forward") => {
            if tokens.len() >= 4 {
                let x: f32 = tokens[1].parse().unwrap();
                let y: f32 = tokens[2].parse().unwrap();
                let z: f32 = tokens[3].parse().unwrap();

                state.forward = [x, y, z];
                state.right = vec3_normalized(vec3_cross(state.forward, state.up));
                state.up = vec3_normalized(vec3_cross(state.right, state.forward));

                println!("forward set to {} {} {}", x, y, z);
                println!(
                    "-- caused right to be set to {} {} {}",
                    state.right[0], state.right[1], state.right[2]
                );
                println!(
                    "-- caused up to be set to {} {} {}",
                    state.up[0], state.up[1], state.up[2]
                );
            } else {
                eprintln!("Invalid eye format");
            }
        }
        Some(&"up") => {
            if tokens.len() >= 4 {
                let x: f32 = tokens[1].parse().unwrap();
                let y: f32 = tokens[2].parse().unwrap();
                let z: f32 = tokens[3].parse().unwrap();

                state.up = vec3_normalized(vec3_cross(state.forward, state.right));

                println!("up set to {} {} {}", x, y, z);
                // println!(
                //     "-- caused right to be set to {} {} {}",
                //     state.right[0], state.right[1], state.right[2]
                // );
            } else {
                eprintln!("Invalid eye format");
            }
        }
        Some(&"roughness") => {
            if tokens.len() >= 2 {
                let sigma: f32 = tokens[1].parse().unwrap();

                state.rand = Some(sigma);

                println!("rand set to {}", sigma);
                // println!(
                //     "-- caused right to be set to {} {} {}",
                //     state.right[0], state.right[1], state.right[2]
                // );
            } else {
                eprintln!("Invalid roughness format");
            }
        }
        Some(&"bounces") => {
            if tokens.len() >= 2 {
                let bounces: u8 = tokens[1].parse().unwrap();
                state.bounces = bounces;

                println!("bounces set to {}", bounces);
            } else {
                eprintln!("Invalid bounces format");
            }
        }
        Some(&"shininess") => {
            if tokens.len() == 2 {
                let value: f32 = tokens[1].parse().unwrap();
                state.shininess = [value, value, value];
                println!("Shininess set to {}, {}, {}", value, value, value);
            } else if tokens.len() == 4 {
                // Case where three parameters are provided
                let sr: f32 = tokens[1].parse().unwrap();
                let sg: f32 = tokens[2].parse().unwrap();
                let sb: f32 = tokens[3].parse().unwrap();
                state.shininess = [sr, sg, sb];
                println!("Shininess set to [{}, {}, {}]", sr, sg, sb);
            } else {
                eprintln!("Invalid bounces format");
            }
        }
        Some(&"ior") => {
            if tokens.len() >= 2 {
                let ior: f32 = tokens[1].parse().unwrap();
                state.ior = ior;

                println!("ior set to {}", ior);
            } else {
                eprintln!("Invalid ior format");
            }
        }
        Some(&"dof") => {
            if tokens.len() >= 3 {
                let focus: f32 = tokens[1].parse().unwrap();
                let lens: f32 = tokens[2].parse().unwrap();
                state.dof = Some([focus, lens]);

                println!("focus set to {}", focus);
                println!("lens set to {}", lens);
            } else {
                eprintln!("Invalid dof format");
            }
        }
        Some(&"aa") => {
            if tokens.len() >= 2 {
                let n: u8 = tokens[1].parse().unwrap();
                state.aa = n;

                println!("aa set to {}", n);
            } else {
                eprintln!("Invalid a format");
            }
        }
        _ => {
            if line.contains("#") || line.trim().is_empty() {
                return;
            }
            eprintln!("Unknown command: {}", line);
        }
    }
}

fn read_input_file(filename: &str, state: &mut RenderState) -> io::Result<()> {
    // Open the file in read-only mode
    let path = Path::new(filename);
    let file = File::open(&path)?;

    let reader = io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        process_line(&line, state);
    }

    state.bvh = Some(build_bvh(state.spheres.clone()));
    println!("BVH constructed from N={}", state.spheres.len());

    if let Some(img) = compute_rays(state) {
        match img.save(&state.filename) {
            Ok(_) => println!("Image saved successfully as {}", state.filename),
            Err(e) => eprintln!("Failed to save image: {}", e),
        }
    } else {
        eprintln!("Failed to compute rays: BVHNode might be missing");
    }

    Ok(())
}

fn main() {
    //base implements png color sphere sun
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input_file>", args[0]);
        return;
    }

    let filename = &args[1];
    println!("Reading {}", filename);

    let mut state = RenderState::new();

    match read_input_file(filename, &mut state) {
        Ok(()) => println!("File parsed successfully"),
        Err(e) => eprintln!("Error reading file: {}", e),
    }
}
