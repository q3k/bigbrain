use std::time;

pub mod types;
pub mod maths;
pub mod proto;
pub mod network;
mod mnist;

use protobuf::Message;

fn main() {
    env_logger::init();

    let now = time::Instant::now();
    let train = mnist::load("train").unwrap();
    let test = mnist::load("t10k").unwrap();
    let elapsed = now.elapsed();
    log::info!("Loaded {} training / {} test images in {:?}.", train.len(), test.len(), elapsed);

    let num_train = (train.len() as f64 * 5.0/6.0) as usize;
    let num_validation = train.len() - num_train;
    log::info!("Split into {} training and {} validation", num_train, num_validation);
    let (train, validation) = train.split(num_train);

    let now = time::Instant::now();
    let size = vec![
        28*28 as usize, 
        30usize,
        10usize,
    ];
    let mut net = network::Network::new(&size);
    let elapsed = now.elapsed();
    log::info!("Created random network {:?} in {:?}.", size, elapsed);
    
    net.sgd(&train, 30, 32, 3.0, Some(&test));
    let proto = net.proto();
    let mut f = std::fs::File::create("net.pb").unwrap();
    proto.write_to_writer(&mut f).unwrap();
}
