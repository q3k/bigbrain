pub trait Input {
    fn size(&self) -> usize;
    fn data(&self) -> &Vec<f32>;
}

pub trait Output {
    fn size(&self) -> usize;
    fn data(&self) -> &Vec<f32>;
    fn from_nn_output(data: Vec<f32>) -> Self;
}

pub trait Data {
    type Input: Input;
    type Output: Output;

    fn iter<'a>(&'a self) -> Box<dyn std::iter::ExactSizeIterator<Item = &'a (Self::Input, Self::Output)> + 'a>;
}

pub struct InMemoryData<I: Input, O: Output> {
    inner: Vec<(I, O)>,
}

impl <I: Input, O: Output> Data for InMemoryData<I, O> {
    type Input = I;
    type Output = O;
    fn iter<'a>(&'a self) -> Box<dyn std::iter::ExactSizeIterator<Item = &'a (I, O)> + 'a> {
        Box::new(self.inner.iter())
    }
}

impl <I: Input, O: Output> InMemoryData<I, O> {
    pub fn split(self, at: usize) -> (Self, Self) {
        let mut vec1 = self.inner;
        let vec2 = vec1.split_off(at);
        (
            Self { inner: vec1, },
            Self { inner: vec2, },
        )
    }
    pub fn new(inner: Vec<(I, O)>) -> Self {
        Self {
            inner
        }
    }
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}
