use wasm_bindgen::prelude::*;

use protobuf::Message;
use bigbrain::types::Output;

macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

use bigbrain::{
    network,
    mnist,
    proto::network as npb,
};

#[wasm_bindgen]
pub struct Brain {
    net: network::Network<mnist::Data>,
}

#[wasm_bindgen]
impl Brain {
    pub fn new(pb: Box<[u8]>) -> Self {
        console_error_panic_hook::set_once();
        let pb = npb::Network::parse_from_bytes(&pb).unwrap();
        Brain {
            net: network::Network::from_proto(&pb).unwrap(),
        }
    }

    pub fn see(&self, data: Box<[u8]>) -> Vec<f32> {
        let vec: Vec<f32> = data.iter().map(|el| {
            ((255 - el) as f32) / 255.0
        }).collect();
        //log!("{:?}", vec);
        let img = mnist::Image::from_vec(&vec);
        let res = self.net.feedforward(&img);
        return res.data().iter().cloned().collect();
    }
}
