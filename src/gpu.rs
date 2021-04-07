use cuda::{Context, Function, DeviceBuffer, CudaError, Module, DevicePtr, Arg};


pub struct CudaModel<'a> {
    pub embeddings_forward: CudaLstm<'a, 100, FORWARD_EMBEDDING>,
    pub embeddings_reverse: CudaLstm<'a, 100, FORWARD_EMBEDDING>,
    pub embeddings2rnn: CudaLinear<'a, EMBEDDINGS, EMBEDDINGS>,
    pub linear: CudaLinear<'a, {2 * RNN_SIZE}, FEATURES>,
    pub rnn: CudaLstm<'a, EMBEDDINGS, RNN_SIZE>,
    pub rnn_reverse: CudaLstm<'a, EMBEDDINGS, RNN_SIZE>,
}

macro_rules! launch {
    ($kernel:expr, x=$x:expr, $(y=$y:expr,)? { $( $name:ident $(= $val:expr)? ,)* } ) => ({
        // assign first, so they are dropped at the end
        $(
            $( let $name = $val; )?
        )*
        let mut args = [ $(
            Arg::borrow(&$name),
        )* ];

        let grid_x = (($x + 31) / 32) as u32;
        let grid_y = 1;
        $(let grid_y = $y as u32;)?
        let block_x = 32;
        unsafe {
            $kernel.launch(
                [grid_x, grid_y, 1],
                [block_x, 1, 1],
                0,
                &mut args[..]
            ).unwrap();
        }
    });
}

impl<'a> CudaModel<'a> {
    pub fn load(model: &Model, context: &'a Context, module: &Module<'a>) -> Result<Self, CudaError> {
        Ok(CudaModel {
            embeddings_forward: CudaLstm::new(&model.embeddings_forward.rnn, context, module)?,
            embeddings_reverse: CudaLstm::new(&model.embeddings_reverse.rnn, context, module)?,
            embeddings2rnn: CudaLinear::new(&model.embedding2nn, context, module)?,
            linear: CudaLinear::new(&*model.linear, context, module)?,
            rnn: CudaLstm::new(&model.rnn, context, module)?,
            rnn_reverse: CudaLstm::new(&model.rnn_reverse, context, module)?,
        })
    }
}

pub struct CudaLinear<'a, const N: usize, const M: usize> where 
[u8; simd(N)]: Sized,
[u8; simd(M)]: Sized,
{
    context: &'a Context,
    linear_func: Function<'a>,
    linear_batch_func: Function<'a>,
    linear_data: DeviceBuffer<'a, f32>,
}
impl<'a, const N: usize, const M: usize> CudaLinear<'a, N, M> where 
[u8; simd(N)]: Sized,
[u8; simd(M)]: Sized,
{
pub fn new<'b>(linear: &Linear<N, M>, context: &'a Context, module: &'b Module<'a>) -> Result<Self, CudaError> {
    let linear_func = module.get("linear")?;
    let linear_batch_func = module.get("linear_batch")?;
    let linear_data = context.create_device_buffer_from(linear.as_ref())?;

    Ok(CudaLinear {
        linear_func, linear_data, context, linear_batch_func
    })
}
pub fn run_ptx<'b, I>(&mut self, inputs: I) -> Result<(), CudaError>
    where I: Iterator<Item=(Vector<N>, &'b mut Vector<M>)>
{
    let mut input_buffer = self.context.create_device_buffer::<f32>(N).unwrap();
    let mut accumulator = self.context.create_device_buffer::<f32>(M).unwrap();

    let data_ptr = self.linear_data.device_ptr();

    let bias_off = 0;
    let weight_off = bias_off + simd_size(M);

    for (input, output) in inputs {
        input_buffer.copy_from(&*input)?;
        accumulator.set_null();
        launch!(self.linear_func, x=M, {
            weight_ptr = data_ptr.offset(weight_off),
            bias_ptr = data_ptr.offset(bias_off),
            input_ptr = input_buffer.device_ptr(),
            accumulator_ptr = accumulator.device_ptr(),
            n = N as u32,
            stride = simd_size(N) as u32,
            m = M as u32,
        });
        accumulator.copy_to(output.as_mut())?;
    }

    Ok(())
}
pub fn run_batched<'b, I>(&mut self, inputs: I) -> Result<(), CudaError>
    where I: Iterator, I::Item: Iterator<Item=(Vector<N>, &'b mut Vector<M>)>
{
    let mut inputs: Vec<_> = inputs.collect();
    let batch_size = inputs.len();
    
    let mut input_buffer = self.context.create_device_buffer::<f32>(batch_size * simd_size(N)).unwrap();
    let mut accumulator = self.context.create_device_buffer::<f32>(batch_size * simd_size(M)).unwrap();

    let data_ptr = self.linear_data.device_ptr();

    let bias_off = 0;
    let weight_off = bias_off + simd_size(M);

    loop {
        let mut any = false;
        let mut outputs = vec![];
        for (i, input) in inputs.iter_mut().enumerate() {
            if let Some((input, output)) = input.next() {
                input_buffer.copy_from_offset(&*input, i * simd_size(N))?;
                outputs.push((i, output));
            }
            any = true;
        }
        if !any {
            break;
        }

        accumulator.set_null();
        launch!(self.linear_batch_func, x=batch_size, y=M, {
            weight_ptr = data_ptr.offset(weight_off),
            bias_ptr = data_ptr.offset(bias_off),
            input_ptr = input_buffer.device_ptr(),
            accumulator_ptr = accumulator.device_ptr(),
            n = N as u32,
            stride = simd_size(N) as u32,
            accumulator_chunk = simd_size(M) as u32,
        });

        for (i, output) in outputs {
            accumulator.copy_to_offset(output.as_mut(), i * simd_size(N))?;
        }
    }

    Ok(())
}
}

pub struct CudaLstm<'a, const N: usize, const M: usize> where 
[u8; simd(N)]: Sized,
[u8; simd(M)]: Sized,
{
context: &'a Context,
linear: Function<'a>,
linear_batch: Function<'a>,
fold: Function<'a>,
fold_batch: Function<'a>,
lstm_data: DeviceBuffer<'a, f32>
}
impl<'a, const N: usize, const M: usize> CudaLstm<'a, N, M> where 
[u8; simd(N)]: Sized,
[u8; simd(M)]: Sized,
{
pub fn new<'b>(lstm: &Lstm<N, M>, context: &'a Context, module: &'b Module<'a>) -> Result<Self, CudaError> {
    let linear = module.get("linear")?;
    let linear_batch = module.get("linear_batch")?;

    let fold = module.get("lstm_fold")?;
    let fold_batch = module.get("lstm_fold_batch")?;
    let lstm_data = context.create_device_buffer_from(lstm.as_ref())?;

    Ok(CudaLstm {
        context, linear, fold, lstm_data, linear_batch, fold_batch
    })
}
pub fn run_ptx<'b, I>(&mut self, inputs: I) -> Result<(), CudaError>
    where I: Iterator<Item=(Option<&'b LstmState<M>>, &'b Vector<N>, Option<&'b mut Vector<M>>)>
{
    let data_ptr = self.lstm_data.device_ptr();

    let off_b_i = 0;
    let off_b_h = off_b_i + 4 * simd_size(M);
    let off_w_i = off_b_h + 4 * simd_size(M);
    let off_w_h = off_w_i + 4 * M * simd_size(N);
    let off_end = off_w_h + 4 * M * simd_size(M);
    assert_eq!(self.lstm_data.len(), off_end);
    
    let mut state_buffer = self.context.create_device_buffer::<f32>(2 * simd_size(M))?;
    let mut input_buffer = self.context.create_device_buffer::<f32>(simd_size(N)).unwrap();
    let mut accumulator = self.context.create_device_buffer::<f32>(4 * M).unwrap();

    let input_ptr = input_buffer.device_ptr();
    let state_h_ptr = state_buffer.device_ptr().offset(simd_size(M));
    let ops = [
        // N x M + M
        (off_w_i, off_b_i, input_ptr, N, simd_size(N)),

        // M x M + M
        (off_w_h, off_b_h, state_h_ptr, M, simd_size(M)),
    ];

    for (state, input, output) in inputs {
        if let Some(state) = state {
            state_buffer.copy_from(state.as_ref())?;
        }
        input_buffer.copy_from(input.buffer())?;
        accumulator.set_null();

        for &(weight_off, bias_off, input_ptr, n, stride) in &ops {
            launch!(self.linear, x=M, {
                weight_ptr = data_ptr.offset(weight_off),
                bias_ptr = data_ptr.offset(bias_off),
                input_ptr = input_ptr,
                accumulator_ptr = accumulator.device_ptr(),
                n = n as u32,
                stride = stride as u32,
                m = 4 * M as u32,
            });
        }

        launch!(self.fold, x=M, {
            data_ptr = accumulator.device_ptr(),
            state_ptr = state_buffer.device_ptr(),
            m = M as u32,
            stride = simd_size(M) as u32,
        });

        if let Some(out) = output {
            let (_, h) = state_buffer.split_at(simd_size(M));
            h.copy_to(out.as_mut())?;
        }
    }

    Ok(())
}
pub fn run_batched<'b, I>(&mut self, inputs: I) -> Result<(), CudaError>
    where I: Iterator,
    I::Item: Iterator<Item=(Option<&'b LstmState<M>>, &'b Vector<N>, Option<&'b mut Vector<M>>)>
{
    let data_ptr = self.lstm_data.device_ptr();

    let off_b_i = 0;
    let off_b_h = off_b_i + 4 * simd_size(M);
    let off_w_i = off_b_h + 4 * simd_size(M);
    let off_w_h = off_w_i + 4 * M * simd_size(N);
    let off_end = off_w_h + 4 * M * simd_size(M);
    assert_eq!(self.lstm_data.len(), off_end);
    
    let mut inputs: Vec<_> = inputs.collect();
    let batch_size = inputs.len();
    let state_chunk = 2 * simd_size(M);
    let input_chunk = simd_size(N);
    let accumulator_chunk = 4 * M;
    let mut state_buffer = self.context.create_device_buffer::<f32>(batch_size * state_chunk)?;
    let mut input_buffer = self.context.create_device_buffer::<f32>(batch_size * input_chunk).unwrap();
    let mut accumulator = self.context.create_device_buffer::<f32>(batch_size * accumulator_chunk).unwrap();

    let input_ptr = input_buffer.device_ptr();
    let state_h_ptr = state_buffer.device_ptr().offset(simd_size(M));
    let ops = [
        // N x M + M
        (off_w_i, off_b_i, input_ptr, N, simd_size(N)),

        // M x M + M
        (off_w_h, off_b_h, state_h_ptr, M, simd_size(M)),
    ];

    loop {
        let mut any = false;
        let mut outputs = vec![];
        for (i, input) in inputs.iter_mut().enumerate() {
            if let Some((state, input, output)) = input.next() {
                if let Some(state) = state {
                    state_buffer.copy_from_offset(state.as_ref(), i * state_chunk)?;
                }
                input_buffer.copy_from_offset(input.buffer(), i * input_chunk)?;
                any = true;
                outputs.push(output);
            }
        }

        if !any {
            break;
        }

        accumulator.set_null();

        for &(weight_off, bias_off, input_ptr, n, stride) in &ops {
            launch!(self.linear_batch, x=batch_size, y=4*M, {
                weight_ptr = data_ptr.offset(weight_off),
                bias_ptr = data_ptr.offset(bias_off),
                input_ptr = input_ptr,
                accumulator_ptr = accumulator.device_ptr(),
                n = n as u32,
                stride = stride as u32,
                m = 4 * M as u32,
                accumulator_chunk,
            });
        }
        /*
        launch!(self.fold_batch, x=M, y=batch_size, {
            data_ptr = accumulator.device_ptr(),
            state_ptr = state_buffer.device_ptr(),
            state_stride = simd_size(M) as u32,
            m = M as u32,
        });
        */
        for (i, output) in outputs.iter_mut().enumerate() {
            if let Some(out) = output {
                state_buffer.copy_to_offset(out.as_mut(), i * state_chunk + simd_size(M))?;
            }
        }
    }

    Ok(())
}
}