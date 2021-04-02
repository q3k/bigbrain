use itertools::Itertools;
use rayon::prelude::*;

use crate::types as t;
use crate::types::{Input, Output};
use crate::proto::network as npb;
use crate::maths::{Vector, Matrix, Shape};

pub struct Network {
    sizes: Vec<usize>,
    biases: Vec<Vector>,
    weights: Vec<Matrix>,
}

impl Network {
    pub fn new(sizes: &Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();

        let biases = sizes[1..].iter().map(|s| Vector::random(*s, &mut rng)).collect();
        let weights = sizes.iter().zip(sizes[1..].iter()).map(|(from, to)| {
            Matrix::random(Shape::new(*to, *from), &mut rng)
        }).collect();

        Self {
            sizes: sizes.clone(),
            biases, weights,
        }
    }

    pub fn proto(&self) -> npb::Network {
        let mut res = npb::Network::new();

        let mut input = npb::InputLayer::new();
        input.set_size(self.weights[0].shape().columns() as u32);
        res.set_input(input);

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let mut inner = npb::InnerLayer::new();
            inner.set_size(b.len() as u32);
            inner.set_biases(b.iter().cloned().collect());
            
            let rows = w.shape().rows();
            let columns = w.shape().columns();
            let mut weights: Vec<f32> = Vec::with_capacity(rows * columns);
            for row in &w.rows {
                for el in row.iter() {
                    weights.push(*el);
                }
            }
            inner.set_weights(weights);
            res.mut_inner().push(inner);
        }

        res
    }

    pub fn dump(&self) {
        for (i, (b, w)) in self.biases.iter().zip(self.weights.iter()).enumerate() {
            log::info!("l {}: biases: {:?}", i, b);
            log::info!("l {}: weights:", i);
            for row in w.rows.iter() {
                log::info!("   {:?}", row);
            }
        }
    }

    fn feedforward<D: t::Data>(&self, i: &D::Input) -> D::Output {
        let mut a = i.data().clone();
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let mut z = w.mult_vec(&a);
            z.add_mut(&b);
            for e in z.iter_mut() {
                *e = sigmoid(*e);
            }
            a = z;
        }
        D::Output::from_nn_output(a)
    }

    pub fn sgd<D: t::Data>(
        &mut self,
        data: &D,
        epochs: usize,
        mini_batch_size: usize,
        eta: f32,
        test_data: Option<&D>,
    ) {
        for j in 0usize..epochs {
            let shuffled = data.shuffle();
            let batches = shuffled.iter().chunks(mini_batch_size);
            for batch in &batches {
                self.update_batch::<D, _>(batch, eta);
            }
            if let Some(td) = test_data {
                let n_test = td.size();
                log::info!("Epoch {}: {} / {}", j, self.evaluate(td), n_test);
            } else {
                log::info!("Epoch {} complete", j);
            }
        }
    }

    fn update_batch<'a, D, B>(
        &mut self,
        batch: B,
        eta: f32,
    ) where 
        D: t::Data,
        D::Input: 'a + Sync,
        D::Output: 'a + Sync,
        B: std::iter::Iterator<Item = &'a (D::Input, D::Output)>,
    {
        let mut nabla_b: Vec<Vector> = self.biases.iter().map(|el| {
            Vector::zeroes(el.len())
        }).collect();
        let mut nabla_w: Vec<Matrix> = self.weights.iter().map(|el| {
            Matrix::zeroes(*el.shape())
        }).collect();

        let batch: Vec<(D::Input, D::Output)> = batch.cloned().collect();
        let par = batch.par_iter().map(|(input, output)| {
            self.backprop::<D>(input, output)
        });

        let mut batch_size = 0usize;
        for (delta_nabla_b, delta_nabla_w) in par.collect::<Vec<_>>() {
            for (nb, dnb) in nabla_b.iter_mut().zip(delta_nabla_b.into_iter()) {
                nb.add_mut(&dnb)
            }
            for (nw, dnw) in nabla_w.iter_mut().zip(delta_nabla_w.into_iter()) {
                nw.add_mut(&dnw)
            }
            batch_size += 1;
        }

        for (w, mut nw) in self.weights.iter_mut().zip(nabla_w.into_iter()) {
            nw.mult_scalar_mut(eta / (batch_size as f32));
            w.sub_mut(&nw)
        }

        for (b, mut nb) in self.biases.iter_mut().zip(nabla_b.into_iter()) {
            nb.mult_scalar_mut(eta / (batch_size as f32));
            b.sub_mut(&nb)
        }
    }

    fn evaluate<D: t::Data>(
        &self,
        test_data: &D,
    ) -> usize {
        let mut count: usize = 0;
        for (input, expected) in test_data.iter() {
            let got = self.feedforward::<D>(input);
            if got.onehot_decode() == expected.onehot_decode() {
                count += 1;
            }
        }
        count
    }

    fn backprop<D: t::Data>(
        &self,
        x: &D::Input,
        y: &D::Output,
    ) -> (Vec<Vector>, Vec<Matrix>) {
        let mut nabla_b: Vec<Vector> = vec![];
        let mut nabla_w: Vec<Matrix> = vec![];


        let mut activations: Vec<Vector> = vec![x.data().clone()];
        let mut activation = &activations[0];
        let mut zs: Vec<Vector> = vec![];

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let mut z = w.mult_vec(activation);
            z.add_mut(b);

            zs.push(z.clone());
            for e in z.iter_mut() {
                *e = sigmoid(*e);
            }
            activations.push(z);
            activation = activations.last().unwrap();
        }

        activations.reverse();
        zs.reverse();

        let sp: Vector = zs[0].iter().map(|el| sigmoid_prime(*el)).collect();
        let mut delta: Vector = self.cost_derivative::<D>(&activations[0], y);
        delta.mult_mut(&sp);

        nabla_b.push(delta.clone());
        nabla_w.push(
            Matrix::from_column(&delta).mult(&Matrix::from_row(activations[1].clone()))
        );

        for l in 1..(self.sizes.len()-1) {
            let z = &zs[l];
            let sp: Vector = z.iter().map(|el| sigmoid_prime(*el)).collect();

            let weights = &self.weights[self.weights.len()-l];
            let weights = weights.transpose();
            delta = weights.mult_vec(&delta).mult(&sp);
            nabla_b.push(delta.clone());
            nabla_w.push(
                Matrix::from_column(&delta).mult(&Matrix::from_row(activations[l+1].clone()))
            );
        }

        nabla_b.reverse();
        nabla_w.reverse();
        (nabla_b, nabla_w)
    }

    fn cost_derivative<D: t::Data>(
        &self,
        output_activations: &Vector,
        y: &D::Output,
    ) -> Vector {
        output_activations.sub(y.data())
    }
}

fn sigmoid(z: f32) -> f32 {
    1. / (1. + (-z).exp())
}

fn sigmoid_prime(z: f32) -> f32 {
    sigmoid(z) * (1. - sigmoid(z))
}

