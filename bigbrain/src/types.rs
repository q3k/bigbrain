use rand::thread_rng;
use rand::seq::SliceRandom;

use crate::maths::Vector;

pub trait Input: std::fmt::Debug + Clone + Sync {
    fn size(&self) -> usize;
    fn data(&self) -> &Vector;
}

pub trait Output: std::fmt::Debug + Clone + Sync {
    fn size(&self) -> usize;
    fn data(&self) -> &Vector;
    fn from_nn_output(data: Vector) -> Self;
    fn onehot_decode(&self) -> usize {
        let mut cur: Option<(f32, usize)> = None;
        for (i, &v) in self.data().iter().enumerate() {
            match cur {
                None => {
                    cur = Some((v, i));
                },
                Some((mv, _)) => {
                    if v > mv {
                        cur = Some((v, i));
                    }
                },
            }
        }
        cur.unwrap().1
    }
}

pub trait Data: Sync {
    type Input: Input;
    type Output: Output;

    fn iter<'a>(&'a self) -> Box<dyn std::iter::ExactSizeIterator<Item = &'a (Self::Input, Self::Output)> + 'a>;
    fn shuffle<'a> (&'a self) -> Box<dyn Data<Input = Self::Input, Output = Self::Output> + 'a>;
    fn size(&self) -> usize;
}

pub struct InMemoryData<I: Input, O: Output> {
    inner: Vec<(I, O)>,
}

pub struct InMemoryDataView<'a, I: Input, O: Output> {
    inner: Vec<&'a (I, O)>,
}

impl <I: Input, O: Output> Data for InMemoryData<I, O> {
    type Input = I;
    type Output = O;
    fn iter<'a>(&'a self) -> Box<dyn std::iter::ExactSizeIterator<Item = &'a (I, O)> + 'a> {
        Box::new(self.inner.iter())
    }
    fn shuffle<'a>(&'a self) -> Box<dyn Data<Input = Self::Input, Output = Self::Output>+'a> {
        Box::new(InMemoryDataView {
            inner: self.inner.iter().collect(),
        })
    }
    fn size(&self) -> usize {
        self.inner.len()
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

impl <'a, I: Input, O: Output> Data for InMemoryDataView<'a, I, O> {
    type Input = I;
    type Output = O;
    fn iter<'b>(&'b self) -> Box<dyn std::iter::ExactSizeIterator<Item = &'b (I, O)> + 'b> {
        let mut shuffled: Vec<&'b (I, O)> = self.inner.iter().map(|el| *el).collect();
        shuffled.shuffle(&mut thread_rng());
        Box::new(shuffled.into_iter())
    }
    fn shuffle<'b>(&'b self) -> Box<dyn Data<Input = Self::Input, Output = Self::Output>+'b> {
        Box::new(InMemoryDataView {
            inner: self.inner.iter().cloned().collect(),
        })
    }
    fn size(&self) -> usize {
        self.inner.len()
    }
}
