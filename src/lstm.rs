use crate::vector::{Vector, Matrix, simd};

#[derive(Clone)]
pub struct LstmState<const M: usize>
    where [u8; simd(M)]: Sized
{
    /// cell state
    pub c: Vector<M>,

    /// output state
    pub h: Vector<M>,
}

impl<const M: usize> LstmState<M> where
    [u8; simd(M)]: Sized
{
    pub fn null() -> Self {
        LstmState {
            c: Vector::null(),
            h: Vector::null()
        }
    }
}
///
/// F: Float type,
/// N: Input size
/// M: Hidden layer size
pub struct Lstm<const N: usize, const M: usize> where
    [u8; simd(N)]: Sized,
    [u8; simd(M)]: Sized,
{

    pub b_ii: Vector<M>,
    pub b_if: Vector<M>,
    pub b_ig: Vector<M>,
    pub b_io: Vector<M>,

    pub b_hi: Vector<M>,
    pub b_hf: Vector<M>,
    pub b_hg: Vector<M>,
    pub b_ho: Vector<M>,

    pub w_ii: Matrix<N, M>,
    pub w_if: Matrix<N, M>,
    pub w_ig: Matrix<N, M>,
    pub w_io: Matrix<N, M>,

    pub w_hi: Matrix<M, M>,
    pub w_hf: Matrix<M, M>,
    pub w_hg: Matrix<M, M>,
    pub w_ho: Matrix<M, M>,
}

impl<const N: usize, const M: usize> Lstm<N, M> where
    [u8; simd(N)]: Sized,
    [u8; simd(M)]: Sized,
{
    pub fn null() -> Box<Self> {
        unsafe {
            // contains only f32s.
            Box::new_zeroed().assume_init()
        }
    }
}

impl<const N: usize, const M: usize> Lstm<N, M> where 
    [u8; simd(N)]: Sized,
    [u8; simd(M)]: Sized,
{
    pub fn step(&self, state: &mut LstmState<M>, x: &Vector<N>) {
        let i = (&self.w_ii * x + &self.b_ii + &self.w_hi * &state.h + &self.b_hi).sigmoid();
        let f = (&self.w_if * x + &self.b_if + &self.w_hf * &state.h + &self.b_hf).sigmoid();
        let g = (&self.w_ig * x + &self.b_ig + &self.w_hg * &state.h + &self.b_hg).tanh();
        let o = (&self.w_io * x + &self.b_io + &self.w_ho * &state.h + &self.b_ho).sigmoid();
        let c = f * &state.c + i * g;

        *state = LstmState {
            c,
            h: o * c.tanh()
        };
    }
}
