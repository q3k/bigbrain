use rand_distr::StandardNormal;

#[derive(Clone,Debug)]
pub struct Vector(pub Vec<f32>);

impl Vector {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn zeroes(size: usize) -> Self {
        Vector(vec![0.0f32; size])
    }
    pub fn random<R: rand::Rng>(size: usize, rng: &mut R) -> Self {
        Vector((0..size).map(|_| rng.sample(StandardNormal)).collect())
    }
    pub fn dot(&self, o: &Self) -> f32 {
        if self.len() != o.len() {
            panic!("Shape mismatch: {} and {}", self.len(), o.len())
        }
        self.0.iter().zip(o.0.iter()).map(|(a, b)| a*b).sum()
    }
    pub fn add_mut(&mut self, o: &Self) {
        if self.len() != o.len() {
            panic!("Shape mismatch: {} and {}", self.len(), o.len())
        }
        for (a, b) in self.0.iter_mut().zip(o.0.iter()) {
            *a += b;
        }
    }
    pub fn add(&self, o: &Self) -> Self {
        let mut res = self.clone();
        res.add_mut(o);
        res
    }
    pub fn sub_mut(&mut self, o: &Self) {
        if self.len() != o.len() {
            panic!("Shape mismatch: {} and {}", self.len(), o.len())
        }
        for (a, b) in self.0.iter_mut().zip(o.0.iter()) {
            *a -= b;
        }
    }
    pub fn sub(&self, o: &Self) -> Self {
        let mut res = self.clone();
        res.sub_mut(o);
        res
    }
    pub fn mult_mut(&mut self, o: &Self) {
        if self.len() != o.len() {
            panic!("Shape mismatch: {} and {}", self.len(), o.len())
        }
        for (a, b) in self.0.iter_mut().zip(o.0.iter()) {
            *a *= b;
        }
    }
    pub fn mult(&self, o: &Self) -> Self {
        let mut res = self.clone();
        res.mult_mut(o);
        res
    }
    pub fn mult_scalar_mut(&mut self, val: f32) {
        for e in self.0.iter_mut() {
            *e *= val;
        }
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<f32> {
        self.0.iter_mut()
    }
    pub fn iter(&self) -> std::slice::Iter<f32> {
        self.0.iter()
    }
}

impl std::iter::FromIterator<f32> for Vector {
    fn from_iter<I: std::iter::IntoIterator<Item=f32>>(iter: I) -> Self {
        Self(Vec::from_iter(iter))
    }
}

impl std::iter::IntoIterator for Vector {
    type Item = f32;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl std::ops::Index<usize> for Vector {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl std::ops::IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct Shape(usize, usize);

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.rows(), self.columns())
    }
}

impl Shape {
    pub fn new(rows: usize, columns: usize) -> Self {
        Self(rows, columns)
    }
    pub fn rows(&self) -> usize {
        self.0
    }
    pub fn columns(&self) -> usize {
        self.1
    }
}

#[derive(Clone,Debug)]
pub struct Matrix {
    shape: Shape,
    pub rows: Vec<Vector>,
}

impl Matrix {
    pub fn zeroes(shape: Shape) -> Self {
        Self {
            shape,
            rows: vec![Vector::zeroes(shape.columns()); shape.rows()],
        }
    }

    pub fn new(rows: Vec<Vector>) -> Self {
        let nrows = rows.len();
        if nrows < 1 {
            panic!("No rows given");
        }
        let ncolumns = rows[0].len();
        if !rows.iter().all(|r| r.len() == ncolumns) {
            panic!("Given vectors are not the same length")
        }
        Self {
            shape: Shape::new(nrows, ncolumns),
            rows,
        }
    }

    pub fn random<R: rand::Rng>(shape: Shape, rng: &mut R) -> Self {
        Self {
            shape,
            rows: vec![Vector::random(shape.columns(), rng); shape.rows()],
        }
    }

    pub fn from_row(row: Vector) -> Self {
        Self {
            shape: Shape::new(1, row.len()),
            rows: vec![row],
        }
    }

    pub fn from_column(column: &Vector) -> Self {
        let rows = column.iter().map(|el| Vector(vec![*el])).collect();
        Self {
            shape: Shape::new(column.len(), 1),
            rows,
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn mult_vec(&self, v: &Vector) -> Vector {
        if v.len() != self.shape().columns() {
            panic!("Shape mismatch: V {}, M {}", v.len(), self.shape())
        }
        self.rows.iter().map(|row| row.dot(v)).collect()
    }

    pub fn column(&self, n: usize) -> Vector {
        if n >= self.shape.columns() {
            panic!("Out of bounds: want column {}, have {}", n, self.shape.columns())
        }

        self.rows.iter().map(|row| row[n]).collect()
    }

    pub fn mult(&self, o: &Matrix) -> Matrix {
        if self.shape().columns() != o.shape().rows() {
            panic!("Shape mismatch: self {}, other {}", self.shape(), o.shape())
        }

        let rows = self.rows.iter().map(|row_a| {
            (0..o.shape().columns()).map(|n| {
                row_a.iter().enumerate().map(|(m, v)| {
                    let v2 = o.rows[m][n];
                    v * v2
                }).sum()
            }).collect()
        }).collect();
        Self {
            shape: Shape(self.shape().rows(), o.shape.columns()),
            rows,
        }
    }

    pub fn add(&self, o: &Self) -> Self {
        if self.shape() != o.shape() {
            panic!("Shape mismatch: self {}, other {}", self.shape(), o.shape())
        }
        Self {
            shape: self.shape().clone(),
            rows: self.rows.iter().zip(o.rows.iter()).map(|(a, b)| a.add(b)).collect(),
        }
    }

    pub fn add_mut(&mut self, o: &Self) {
        if self.shape() != o.shape() {
            panic!("Shape mismatch: self {}, other {}", self.shape(), o.shape())
        }
        for (ra, rb) in self.rows.iter_mut().zip(o.rows.iter()) {
            ra.add_mut(rb);
        }
    }
    pub fn sub_mut(&mut self, o: &Self) {
        if self.shape() != o.shape() {
            panic!("Shape mismatch: self {}, other {}", self.shape(), o.shape())
        }
        for (ra, rb) in self.rows.iter_mut().zip(o.rows.iter()) {
            ra.sub_mut(rb);
        }
    }
    pub fn mult_scalar_mut(&mut self, val: f32) {
        for row in self.rows.iter_mut() {
            row.mult_scalar_mut(val);
        }
    }

    pub fn transpose(&self) -> Self {
        let rows = (0..self.shape().columns()).map(|cnum| {
            self.column(cnum)
        }).collect();
        Self {
            shape: Shape::new(self.shape().columns(), self.shape().rows()),
            rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_column_column_mult() {
        let m1 = Matrix::from_column(&Vector(vec![1.,2.,3.]));
        let m2 = Matrix::from_row(Vector(vec![2.,3.,4.]));
        let res = m1.mult(&m2);
        assert_eq!(res.rows[0].0, vec![ 2.0, 3.0,  4.0 ]);
        assert_eq!(res.rows[1].0, vec![ 4.0, 6.0,  8.0 ]);
        assert_eq!(res.rows[2].0, vec![ 6.0, 9.0, 12.0 ]);
    }

    #[test]
    fn test_matrix_matrix_mult() {
        let m1 = Matrix::new(vec![
            Vector(vec![ 0.0,  4.0, -2.0 ]),
            Vector(vec![-4.0, -3.0,  0.0 ]),
        ]);
        let m2 = Matrix::new(vec![
            Vector(vec![ 0.0,  1.0 ]),
            Vector(vec![ 1.0, -1.0 ]),
            Vector(vec![ 2.0,  3.0 ]),
        ]);

        let res = m1.mult(&m2);
        assert_eq!(res.rows[0].0, vec![  0.0, -10.0 ]);
        assert_eq!(res.rows[1].0, vec![ -3.0,  -1.0 ]);
    }

    #[test]
    fn test_matrix_vector_mult() {
        let m1 = Matrix::new(vec![
            Vector(vec![ 1.0, -1.0, 2.0 ]),
            Vector(vec![ 0.0, -3.0, 1.0 ]),
        ]);
        let v2 = Vector(vec![2.0, 1.0, 0.0]);
        let res = m1.mult_vec(&v2);
        assert_eq!(res.0, vec![1.0, -3.0]);
    }
}
