use simd_linalg::{Vector, Matrix, simd, simd_size, ZeroInit, zero};

#[derive(Clone)]
#[repr(C)]
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
unsafe impl<const M: usize> ZeroInit for LstmState<M>
where [u8; simd(M)]: Sized {}

impl<const M: usize> AsRef<[f32]> for LstmState<M> where
    [u8; simd(M)]: Sized,
{
    fn as_ref(&self) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(self as *const _ as *const f32, std::mem::size_of::<Self>() / 4)
        }
    }
}
impl<const M: usize> AsMut<[f32]> for LstmState<M> where
    [u8; simd(M)]: Sized,
{
    fn as_mut(&mut self) -> &mut [f32] {
        unsafe {
            std::slice::from_raw_parts_mut(self as *mut _ as *mut f32, std::mem::size_of::<Self>() / 4)
        }
    }
}

#[repr(C)]
pub struct Gates<T> {
    pub i: T,
    pub f: T,
    pub g: T,
    pub o: T,
}
unsafe impl<T: ZeroInit> ZeroInit for Gates<T> {}

impl<T> Gates<T> {
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self as *const _ as _, 4)
        }
    }
}

///
/// F: Float type,
/// N: Input size
/// M: Hidden layer size
#[repr(C)]
pub struct Lstm<const N: usize, const M: usize> where
    [u8; simd(N)]: Sized,
    [u8; simd(M)]: Sized,
{
    pub b_i: Gates<Vector<M>>,
    pub b_h: Gates<Vector<M>>,
    pub w_i: Gates<Matrix<N, M>>,
    pub w_h: Gates<Matrix<M, M>>,
}

unsafe impl<const N: usize, const M: usize> ZeroInit for Lstm<N, M> where
    [u8; simd(N)]: Sized, [u8; simd(M)]: Sized {}

impl<const N: usize, const M: usize> Lstm<N, M> where 
    [u8; simd(N)]: Sized,
    [u8; simd(M)]: Sized,
{
    pub fn step(&self, state: &mut LstmState<M>, x: &Vector<N>) {
        let i = (&self.w_i.i * x + &self.b_i.i + &self.w_h.i * &state.h + &self.b_h.i).sigmoid();
        let f = (&self.w_i.f * x + &self.b_i.f + &self.w_h.f * &state.h + &self.b_h.f).sigmoid();
        let g = (&self.w_i.g * x + &self.b_i.g + &self.w_h.g * &state.h + &self.b_h.g).tanh();
        let o = (&self.w_i.o * x + &self.b_i.o + &self.w_h.o * &state.h + &self.b_h.o).sigmoid();
        let c = f * &state.c + i * g;

        *state = LstmState {
            c,
            h: o * c.tanh()
        };
    }

    pub fn run<'b, I>(&self, inputs: I)
    where I: Iterator<Item=(Option<&'b LstmState<M>>, &'b Vector<N>, Option<&'b mut Vector<M>>)>
    {
        let mut state = zero::<LstmState<M>>();
        for (new_state, input, output) in inputs {
            if let Some(new_state) = new_state {
                state = new_state.clone();
            }
            self.step(&mut state, input);
            if let Some(output) = output {
                *output = state.h;
            }
        }
    }
}
impl<const N: usize, const M: usize> AsRef<[f32]> for Lstm<N, M> where 
    [u8; simd(N)]: Sized,
    [u8; simd(M)]: Sized,
{
    fn as_ref(&self) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(self as *const _ as *const f32, std::mem::size_of::<Self>() / 4)
        }
    }
}

/// Linear projection with bias
#[derive(Clone, Debug)]
#[repr(C)]
pub struct Linear<const N: usize, const M: usize>
    where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    pub bias: Vector<M>,
    pub weight: Matrix<N, M>
}

unsafe impl<const N: usize, const M: usize> ZeroInit for Linear<N, M>
    where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized {}

impl<const N: usize, const M: usize> Linear<N, M>
    where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    /// create a `Box<Self>` full of zeros
    #[cfg(feature="alloc")]
    pub fn null() -> Box<Self> {
        unsafe {
            Box::new_zeroed().assume_init()
        }
    }

    /// `x -> self.weight * x + self.bias`
    pub fn transform(&self, x: &Vector<N>) -> Vector<M> {
        &self.weight * x + self.bias
    }

    pub fn run<'b, I>(&self, inputs: I)
    where I: Iterator<Item=(Vector<N>, &'b mut Vector<M>)>
    {
        inputs.for_each(|(input, output)| *output = self.transform(&input));
    }
}
impl<const N: usize, const M: usize> AsRef<[f32]> for Linear<N, M>
    where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    fn as_ref(&self) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(self as *const _ as *const f32, std::mem::size_of::<Self>() / 4)
        }
    }
}
impl<const N: usize, const M: usize> AsMut<[f32]> for Linear<N, M>
    where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    fn as_mut(&mut self) -> &mut [f32] {
        unsafe {
            std::slice::from_raw_parts_mut(self as *mut _ as *mut f32, std::mem::size_of::<Self>() / 4)
        }
    }
}
