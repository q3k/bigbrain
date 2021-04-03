use std::time;

use bigbrain::{mnist,network,proto::network as npb};

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

    //let now = time::Instant::now();
    //let size = vec![
    //    28*28 as usize, 
    //    30usize,
    //    10usize,
    //];
    //let mut net = network::Network::new(&size);
    //let elapsed = now.elapsed();
    //log::info!("Created random network {:?} in {:?}.", size, elapsed);
    //net.sgd(&train, 30, 32, 3.0, Some(&test));
    //let proto = net.proto();
    //let mut f = std::fs::File::create("net.pb").unwrap();
    //proto.write_to_writer(&mut f).unwrap();

    let nbytes = std::fs::read("net.pb").unwrap();
    let n = npb::Network::parse_from_bytes(&nbytes).unwrap();
    let net = network::Network::from_proto(&n).unwrap();
    let res = net.evaluate(&test);
    log::info!("{}", res);
}
