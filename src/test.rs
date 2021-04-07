
use crate::vector::*;
use crate::lstm::*;

pub fn init_lstm<const N: usize, const M: usize, I>(mut iter: I) -> Box<Lstm<N, M>>
where
    [u8; simd(N)]: Sized,
    [u8; simd(M)]: Sized,
    I: Iterator<Item=f32>
{
    let mut lstm = zero_box::<Lstm<N, M>>();
    lstm.b_i.as_mut_slice().iter_mut().for_each(|v| v.fill(&mut iter));
    lstm.b_h.as_mut_slice().iter_mut().for_each(|v| v.fill(&mut iter));

    lstm.w_i.as_mut_slice().iter_mut().for_each(|m| m.fill(&mut iter));
    lstm.w_h.as_mut_slice().iter_mut().for_each(|m| m.fill(&mut iter));
    
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

#[cfg(feature="gpu")]
#[test]
fn test_cuda() {
    use std::iter::once;
    use cuda::*;
    
    const MODULE: &[u8] = include_bytes!("../../ptx_linalg/ptx_linalg.cubin");
    let context = Device::get(0).unwrap().create_context().unwrap();
    let module = context.create_module(MODULE).unwrap();

    let mut input = Vector::from(&[0.0; 2148]);
    input[1] = 1.0;
    let mut output_gpu = zero::<Vector<4>>();

    let mut linear = zero::<Linear<2148, 4>>();
    linear.bias[0] = 1.0;
    linear.weight[1][1] = 2.0;
    let mut cuda_linear = CudaLinear::new(&linear, &context, &module).unwrap();

    cuda_linear.run_ptx(once((input, &mut output_gpu)));

    let output_cpu = linear.transform(&input);
    let delta = output_gpu - output_cpu;
    assert!(delta.dot(&delta) < 1e-3, "incorrect result:\ngpu: {}\ncpu: {}", output_gpu, output_cpu);


}