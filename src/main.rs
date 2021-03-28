use std::time;
use rand::Rng;
use rayon::prelude::*;
use crate::types as t;
use crate::types::{Data, Input, Output};

pub mod types;
mod mnist;

struct Network {
    sizes: Vec<usize>,
    biases: Vec<Vec<f32>>,
    weights: Vec<Vec<Vec<f32>>>,
}

impl Network {
    fn new(sizes: &Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();

        let biases = sizes[1..].iter().map(|s| {
            (0..s.clone()).map(|_| rng.gen::<f32>()).collect()
        }).collect();

        let weights = sizes.iter().zip(sizes[1..].iter()).map(|(from, to)| {
            (0..to.clone()).map(|_| {
                (0..from.clone()).map(|_| rng.gen::<f32>()).collect()
            }).collect()
        }).collect();

        Self {
            sizes: sizes.clone(),
            biases, weights,
        }
    }

    fn feedforward<D: t::Data>(&self, i: &D::Input) -> D::Output {
        let a = i.data();
        let o = self.biases.iter()
            .zip(self.weights.iter())
            .fold(a.clone(), |a, (biases, weights)| {
                mat_vec_mult(weights, &a).iter()
                    .zip(biases.iter())
                    .map(|(aa, b)| {
                        sigmoid(aa+b)
                    }).collect()
            });
        D::Output::from_nn_output(o)
    }
}

fn sigmoid(z: f32) -> f32 {
    1. / (1. + (-z).exp())
}

fn mat_vec_mult(m: &Vec<Vec<f32>>, v: &Vec<f32>) -> Vec<f32> {
    m.iter().enumerate().map(|(i, row)| {
        if row.len() != v.len() {
            panic!("Mismatch: V {}, M {}x{}", v.len(), m.len(), row.len());
        }
        row.iter().zip(v.iter()).map(|(rr, vv)| rr * vv).sum()
    }).collect()
}

fn main() {
    env_logger::init();

    let now = time::Instant::now();
    let train = mnist::load("train").unwrap();
    let elapsed = now.elapsed();
    log::info!("Loaded {} images in {:?}.", train.len(), elapsed);

    let num_train = (train.len() as f64 * 5.0/6.0) as usize;
    let num_validation = train.len() - num_train;
    log::info!("Split into {} training and {} validation", num_train, num_validation);
    let (train, validation) = train.split(num_train);

    let now = time::Instant::now();
    let size = vec![
        28*28 as usize, 
        15usize,
        10usize,
    ];
    let network = Network::new(&size);
    let elapsed = now.elapsed();
    log::info!("Created random network {:?} in {:?}.", size, elapsed);

    let now = time::Instant::now();
    let data = train.iter().next().unwrap();
    let res = network.feedforward::<mnist::Data>(&data.0);
    let elapsed = now.elapsed();
    log::info!("feedforward: {:?} in {:?}", res, elapsed);
}
