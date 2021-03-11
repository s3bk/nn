
use crate::vector::*;
use crate::lstm::*;

pub fn init_lstm<const N: usize, const M: usize, I>(mut iter: I) -> Box<Lstm<N, M>>
where
    [u8; simd(N)]: Sized,
    [u8; simd(M)]: Sized,
    I: Iterator<Item=f32>
{
    let mut lstm = Lstm::null();
    lstm.b_ii.fill(&mut iter);
    lstm.b_if.fill(&mut iter);
    lstm.b_ig.fill(&mut iter);
    lstm.b_io.fill(&mut iter);
    lstm.b_hi.fill(&mut iter);
    lstm.b_hf.fill(&mut iter);
    lstm.b_hg.fill(&mut iter);
    lstm.b_ho.fill(&mut iter);

    lstm.w_ii.fill(&mut iter);
    lstm.w_if.fill(&mut iter);
    lstm.w_ig.fill(&mut iter);
    lstm.w_io.fill(&mut iter);
    lstm.w_hi.fill(&mut iter);
    lstm.w_hf.fill(&mut iter);
    lstm.w_hg.fill(&mut iter);
    lstm.w_ho.fill(&mut iter);
    
    lstm
}

pub struct Lstm100x1024 {
    lstm: Box<Lstm<100, 1024>>,
    state: Box<LstmState<1024>>,
    x: Box<Vector<100>>,
}
impl Lstm100x1024 {
    pub fn new() -> Self {
        use rand::distributions::{Distribution, Standard};

        let rng = rand::thread_rng();
        let dist = Standard;
        let mut iter = dist.sample_iter(rng);

        let lstm = init_lstm::<100, 1024, _>(&mut iter);

        let mut x = Box::new(Vector::null());
        x.fill(&mut iter);

        Lstm100x1024 {
            lstm,
            state: Box::new(LstmState::null()),
            x
        }
    }
    pub fn step(&mut self) {
        self.lstm.step(&mut self.state, &self.x);
    }
}
