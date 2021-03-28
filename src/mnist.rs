use std::{io, fs};
use byteorder::{
    BigEndian,
    ReadBytesExt,
};
use flate2::read::GzDecoder;

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
    pixels: Vec<u8>,
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
        Ok(Image {
            pixels, width, height,
        })
    }
}

impl Into<Vec<f32>> for &Image {
    fn into(self) -> Vec<f32> {
        self.pixels.iter().map(|px| {
            (*px as f32) / 255.0
        }).collect()
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

pub struct LabelFile {
    magic: u32,
    num_labels: u32,
    labels: Vec<u8>,
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
            magic, num_labels, labels,
        })
    }
}

pub struct Set {
    pub images: ImageFile,
    labels: LabelFile,
}

impl Set {
    pub fn load(name: &str) -> Result<Self> {
        let labels_gz = fs::File::open(format!("{}-labels-idx1-ubyte.gz", name))?;
        let labels = LabelFile::parse(GzDecoder::new(labels_gz))?;
        let images_gz = fs::File::open(format!("{}-images-idx3-ubyte.gz", name))?;
        let images = ImageFile::parse(GzDecoder::new(images_gz))?;
        Ok(Set {
            images, labels,
        })
    }
}
