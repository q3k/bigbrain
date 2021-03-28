use std::{io, fs};
use byteorder::{
    BigEndian,
    ReadBytesExt,
};
use flate2::read::GzDecoder;
use crate::types as t;

#[derive(Debug)]
pub enum Error {
    IO(io::Error),
    InvalidMagic,
    Parse(String),
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::IO(err)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub struct Image {
    pixels: Vec<f32>,
    width: usize,
    height: usize,
}

impl Image {
    pub fn parse(
        rdr: &mut impl io::Read,
        width: usize,
        height: usize,
    ) -> Result<Self> {
        let npixels: usize = width * height;
        let mut pixels: Vec<u8> = vec![0u8; npixels];
        rdr.read(&mut pixels)?;
        let pixels: Vec<f32> = pixels.into_iter().map(|u| (u as f32) / 255.0).collect();
        Ok(Image {
            pixels, width, height,
        })
    }
}

impl t::Input for Image {
    fn size(&self) -> usize {
        self.width * self.height
    }
    fn data(&self) -> &Vec<f32> {
        &self.pixels
    }
}

pub struct ImageFile {
    pub images: Vec<Image>,
}

impl ImageFile {
    pub fn parse(mut rdr: impl io::Read) -> Result<Self> {
        let magic = rdr.read_u32::<BigEndian>()?;
        if magic != 2051 {
            return Err(Error::InvalidMagic);
        }
        let num_images = rdr.read_u32::<BigEndian>()?;
        let num_rows = rdr.read_u32::<BigEndian>()?;
        let num_columns = rdr.read_u32::<BigEndian>()?;

        let mut images: Vec<Image> = Vec::with_capacity(num_images as usize);
        for _ in 0..num_images {
            let image = Image::parse(&mut rdr, num_rows as usize, num_columns as usize)?;
            images.push(image);
        }
        Ok(ImageFile {
            images,
        })
    }
}

#[derive(Debug)]
pub struct RecognizedDigit(Vec<f32>);

impl RecognizedDigit {
    pub fn parse(label: usize, max: usize) -> Self {
        let mut data: Vec<f32> = vec![0f32; max];
        data[label] = 1.0f32;
        Self(data)
    }
}

impl t::Output for RecognizedDigit {
    fn size(&self) -> usize {
        self.0.len()
    }
    fn data(&self) -> &Vec<f32> {
        &self.0
    }
    fn from_nn_output(data: Vec<f32>) -> Self {
        if data.len() != 10 {
            panic!("invalid nn output size (got {}, wanted {})", data.len(), 10)
        }
        Self(data)
    }
}

pub struct LabelFile {
    labels: Vec<RecognizedDigit>,
}

impl LabelFile {
    pub fn parse(mut rdr: impl io::Read) -> Result<Self> {
        let magic = rdr.read_u32::<BigEndian>()?;
        if magic != 2049 {
            return Err(Error::InvalidMagic);
        }
        let num_labels = rdr.read_u32::<BigEndian>()?;
        let mut labels: Vec<u8> = vec![0u8; num_labels as usize];
        rdr.read(&mut labels)?;
        Ok(LabelFile {
            labels: labels.into_iter().map(|l| RecognizedDigit::parse(l as usize, 10usize)).collect()
        })
    }
}

pub type Data = t::InMemoryData<Image, RecognizedDigit>;

pub fn load(name: &str) -> Result<Data> {
    let labels_gz = fs::File::open(format!("{}-labels-idx1-ubyte.gz", name))?;
    let labels = LabelFile::parse(GzDecoder::new(labels_gz))?;
    let images_gz = fs::File::open(format!("{}-images-idx3-ubyte.gz", name))?;
    let images = ImageFile::parse(GzDecoder::new(images_gz))?;
    
    let zipped: Vec<(Image, RecognizedDigit)> = images.images.into_iter().zip(labels.labels.into_iter()).collect();
    Ok(t::InMemoryData::new(zipped))
}

