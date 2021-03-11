use ndarray::{Array1, Array2, Ix1, Ix2, ArrayView, ArrayBase, DataOwned, Dimension};
use ndarray_npy::{NpzReader, ReadNpzError, ReadableElement};
use std::io::{Read, Seek};
use crate::lstm::Lstm;
use crate::vector::*;
use std::error::Error;
use std::collections::HashMap;

fn get<S, D, R: Read + Seek>(npz: &mut NpzReader<R>, name: &str) -> Result<ArrayBase<S, D>, Box<dyn Error>>
where
    S::Elem: ReadableElement,
    S: DataOwned,
    D: Dimension,
{
    match npz.by_name(name) {
        Ok(val) => Ok(val),
        Err(ReadNpzError::Zip(e)) => {
            error!("reading {}: {:?}", name, &e);
            Err(e.into())
        }
        Err(e) => {
            error!("reading {}: {:?}", name, &e);
            Err(e.into())
        }
    }
}

macro_rules! load {
    ($npz:ident => $($name:ident : $ty:ty),*) => {
        $(
            let $name: $ty = get(&mut $npz, concat!(stringify!($name), ".npy"))?;
        )*
    }
}

macro_rules! name {
    ($name0:tt $(, $name:tt)*) => (concat!($name0, $("::", $name,)*))
}
macro_rules! path {
    ($name0:tt $(, $name:tt)*) => (concat!($name0, $("_", $name,)* ".npy"))
}

macro_rules! load_lstm {
    ($npz:ident => $($name:tt),* $(: $suffix:tt)?) => ({
        info!(concat!("loading LSTM ", name!($($name),*)));
        let bias_ih: Array1<f32> = get(&mut $npz, path!($($name,)* "bias_ih" $(,$suffix)?))?;
        let bias_hh: Array1<f32> = get(&mut $npz, path!($($name,)* "bias_hh" $(,$suffix)?))?;
        let weight_ih: Array2<f32> = get(&mut $npz, path!($($name,)* "weight_ih" $(,$suffix)?))?;
        let weight_hh: Array2<f32> = get(&mut $npz, path!($($name,)* "weight_hh" $(,$suffix)?))?;
        load_lstm(bias_ih, bias_hh, weight_ih, weight_hh)
    })
}

macro_rules! load_embedding {
    ($npz:ident => $($name:tt),*) => ({
        info!(concat!("loading Embedding ", name!($($name),*)));
        let chars: Array1<u8> = get(&mut $npz, path!($($name,)* "chars"))?;
        let encoder: Array2<f32> = get(&mut $npz, path!($($name,)* "encoder"))?;
        let rnn = load_lstm!($npz => $($name,)* "rnn");
        load_embedding(chars, encoder, rnn)
    });
}

macro_rules! load_linear {
    ($npz:ident => $($name:tt),*) => ({
        info!(concat!("loading Linear ", name!($($name),*)));
        let bias: Array1<f32> = get(&mut $npz, path!($($name,)* "bias"))?;
        let weight: Array2<f32> = get(&mut $npz, path!($($name,)* "weight"))?;
        load_linear(bias, weight)
    });
}

fn unpack_bias<const M: usize>(mut out: [&mut Vector<M>; 4], bias: Array1<f32>) where
    [u8; simd(M)]: Sized
{
    assert_eq!(bias.shape(), &[4 * M]);
    let bias = bias.into_shape((4, M)).unwrap();
    for (out, lane) in out.iter_mut().zip(bias.outer_iter()) {
        out.fill(lane.iter().cloned());
    }
}
fn unpack_weight<const N: usize, const M: usize>(mut out: [&mut Matrix<N, M>; 4], weight: Array2<f32>) where
    [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    assert_eq!(weight.shape(), &[4 * M, N]);
    let weight = weight.into_shape((4, M, N)).unwrap();
    for (out, arr) in out.iter_mut().zip(weight.outer_iter()) {
        out.fill(arr.iter().cloned());
    }
}

fn load_lstm<const N: usize, const M: usize>(
    bias_ih: Array1<f32>, bias_hh: Array1<f32>,
    weight_ih: Array2<f32>, weight_hh: Array2<f32>) -> Box<Lstm<N, M>>
where
    [u8; simd(N)]: Sized,
    [u8; simd(M)]: Sized,
{
    debug!("{:?}", (bias_ih.shape(), bias_hh.shape(), weight_ih.shape(), weight_hh.shape()));

    let mut lstm = Lstm::null();
    // bias_ih => (b_ii|b_if|b_ig|b_io)
    unpack_bias([&mut lstm.b_ii, &mut lstm.b_if, &mut lstm.b_ig, &mut lstm.b_io], bias_ih);
    unpack_bias([&mut lstm.b_hi, &mut lstm.b_hf, &mut lstm.b_hg, &mut lstm.b_ho], bias_hh);
    unpack_weight([&mut lstm.w_ii, &mut lstm.w_if, &mut lstm.w_ig, &mut lstm.w_io], weight_ih);
    unpack_weight([&mut lstm.w_hi, &mut lstm.w_hf, &mut lstm.w_hg, &mut lstm.w_ho], weight_hh);

    lstm
}

fn load_linear<const N: usize, const M: usize>(bias: Array1<f32>, weight: Array2<f32>) -> Box<Linear<N, M>>
where
    [u8; simd(N)]: Sized,
    [u8; simd(M)]: Sized,
{
    assert_eq!(bias.shape(), &[M]);
    assert_eq!(weight.shape(), &[M, N]);

    let mut l = Linear::null();
    l.bias.fill(bias.iter().cloned());
    l.weight.fill(weight.iter().cloned());
    //l.weight.fill(weight.gencolumns().into_iter().flat_map(|c| c.into_iter()).cloned());
    l
}

fn vector<'a, const N: usize>(data: ArrayView<'a, f32, Ix1>) -> Vector<N> where
    [u8; simd(N)]: Sized
{
    assert_eq!(data.shape(), &[N]);
    let mut v = Vector::null();
    v.fill(data.iter().cloned());
    v
}

fn matrix<'a, const N: usize, const M: usize>(data: ArrayView<'a, f32, Ix2>) -> Box<Matrix<N, M>> where
    [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    let mut m = Matrix::null();
    assert_eq!(data.shape(), &[M, N]);
    m.fill(data.iter().cloned());
    m
}

pub struct Embedding<const N: usize, const M: usize>
    where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    pub chars: HashMap<char, Vector<N>>,
    pub rnn: Box<Lstm<N, M>>,
}

fn load_embedding<const N: usize, const M: usize>(keys: Array1<u8>, encoder: Array2<f32>, rnn: Box<Lstm<N, M>>) -> Embedding<N, M>
    where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    assert_eq!(encoder.shape()[1], N);

    let chars = keys.as_slice().unwrap().split(|&b| b == 0xFF)
        .map(|s| {
            if s == b"<unk>" {
                '�'
            } else {
                std::str::from_utf8(s).unwrap().chars().next().unwrap()
            }
        })
        .zip(encoder.outer_iter().map(|v| vector::<N>(v)))
        .collect();

    Embedding { chars, rnn }
}

fn load_tags(data: Array1<u8>) -> Vec<Box<str>> {
    let mut tags = Vec::with_capacity(FEATURES);
    tags.extend(data.as_slice().unwrap().split(|&b| b == 0xFF)
        .map(|s| std::str::from_utf8(s).unwrap().to_owned().into_boxed_str())
    );
    assert_eq!(tags.len(), FEATURES);
    tags
}

pub const FORWARD_EMBEDDING: usize = 1024;
pub const REVERSE_EMBEDDING: usize = 1024;
pub const GLOVE_EMBEDDING: usize = 100;
pub const FEATURES: usize = 20;
pub const RNN_SIZE: usize = 256;

pub const EMBEDDINGS: usize = GLOVE_EMBEDDING + FORWARD_EMBEDDING + REVERSE_EMBEDDING;

pub struct Model {
    pub rnn: Box<Lstm<EMBEDDINGS, RNN_SIZE>>,
    pub rnn_reverse: Box<Lstm<EMBEDDINGS, RNN_SIZE>>,
    pub embeddings_forward: Embedding<100, FORWARD_EMBEDDING>,
    pub embeddings_reverse: Embedding<100, REVERSE_EMBEDDING>,
    pub embeddings_glove: HashMap<String, Vector<GLOVE_EMBEDDING>>,
    pub embedding2nn: Box<Linear<EMBEDDINGS, EMBEDDINGS>>,
    pub transitions: Box<Matrix<FEATURES, FEATURES>>,
    pub linear: Box<Linear<{2 * RNN_SIZE}, FEATURES>>,
    pub tags: Vec<Box<str>>,
}
impl Model {
    pub fn load(reader: impl Read + Seek) -> Result<Model, Box<dyn Error>> {
        let mut npz = NpzReader::new(reader)?;

        let embeddings_forward = load_embedding!(npz => "embeddings_forward");
        let embeddings_reverse = load_embedding!(npz => "embeddings_reverse");
        let rnn = load_lstm!(npz => "rnn");
        let rnn_reverse = load_lstm!(npz => "rnn" : "reverse");

        let embedding2nn = load_linear!(npz => "embedding2nn");
        let linear = load_linear!(npz => "linear");
        
        load![npz => 
            embeddings_glove_words: Array1<u8>,
            embeddings_glove_vectors: Array2<f32>,
            tag_dictionary: Array1<u8>,
            transitions: Array2<f32>
        ];

        let embeddings_glove = embeddings_glove_words.as_slice().unwrap()
            .split(|&b| b == 0xff)
            .zip(embeddings_glove_vectors.outer_iter())
            .map(|(bytes, view)| (
                std::str::from_utf8(bytes).unwrap().into(),
                vector::<100>(view)
            )).collect();

        Ok(Model {
            rnn,
            rnn_reverse,
            embeddings_forward,
            embeddings_reverse,
            embeddings_glove,
            embedding2nn,
            transitions: matrix(transitions.view()),
            linear,
            tags: load_tags(tag_dictionary)
        })
    }
}
