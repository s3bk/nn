use ndarray::{Array1, Array2, Ix1, Ix2, ArrayView, ArrayBase, DataOwned, Dimension};
use ndarray_npy::{NpzReader, ReadNpzError, ReadableElement};
use std::io::{Read, Seek};
use crate::lstm::{Lstm, Gates, LstmState, Linear};
use simd_linalg::*;
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

fn unpack_bias<const M: usize>(out: &mut Gates<Vector<M>>, bias: Array1<f32>) where
    [u8; simd(M)]: Sized
{
    assert_eq!(bias.shape(), &[4 * M]);
    let bias = bias.into_shape((4, M)).unwrap();
    for (out, lane) in out.as_mut_slice().iter_mut().zip(bias.outer_iter()) {
        out.fill(lane.iter().cloned());
    }
}
fn unpack_weight<const N: usize, const M: usize>(out: &mut Gates<Matrix<N, M>>, weight: Array2<f32>) where
    [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    assert_eq!(weight.shape(), &[4 * M, N]);
    let weight = weight.into_shape((4, M, N)).unwrap();
    for (out, arr) in out.as_mut_slice().iter_mut().zip(weight.outer_iter()) {
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
    
    let mut lstm = zero_box::<Lstm<N, M>>();
    unpack_bias(&mut lstm.b_i, bias_ih);
    unpack_bias(&mut lstm.b_h, bias_hh);
    unpack_weight(&mut lstm.w_i, weight_ih);
    unpack_weight(&mut lstm.w_h, weight_hh);

    lstm
}

fn load_linear<const N: usize, const M: usize>(bias: Array1<f32>, weight: Array2<f32>) -> Box<Linear<N, M>>
where
    [u8; simd(N)]: Sized,
    [u8; simd(M)]: Sized,
{
    assert_eq!(bias.shape(), &[M]);
    assert_eq!(weight.shape(), &[M, N]);

    let mut l = zero_box::<Linear<N, M>>();
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
    let mut m = zero_box::<Matrix<N, M>>();
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

    pub fn forward_embedding(&self, tokens: &[&str], out: &mut[Vector<FORWARD_EMBEDDING>]) {
        let embedding = &self.embeddings_forward;
        let mut state = LstmState::null();
        let start_marker = '\n';
        let end_marker = ' ';
        let v_start = &embedding.chars[&start_marker];
        let v_end = &embedding.chars[&end_marker];

        embedding.rnn.step(&mut state, v_start);
        for (&token, out) in tokens.iter().zip(out) {
            for c in token.chars() {
                if let Some(v) = embedding.chars.get(&c) {
                    embedding.rnn.step(&mut state, v);
                }
            }
            embedding.rnn.step(&mut state, v_end);
            *out = state.h;
        }
    }

    pub fn reverse_embedding(&self, tokens: &[&str], out: &mut [Vector<REVERSE_EMBEDDING>]) {
        let embedding = &self.embeddings_reverse;
        let mut state = LstmState::null();
        let start_marker = '\n';
        let end_marker = ' ';
        let v_start = &embedding.chars[&start_marker];
        let v_end = &embedding.chars[&end_marker];

        embedding.rnn.step(&mut state, v_start);
        for (&token, out) in tokens.iter().zip(out).rev() {
            for c in token.chars().rev() {
                if let Some(v) = embedding.chars.get(&c) {
                    embedding.rnn.step(&mut state, v);
                }
            }
            embedding.rnn.step(&mut state, v_end);
            *out = state.h;
        }
    }
    pub fn glove_embedding(&self, tokens: &[&str], out: &mut [Vector<GLOVE_EMBEDDING>]) {
        let embedding = &self.embeddings_glove;

        for (&token, out) in tokens.iter().zip(out) {
            if let Some(v) = embedding.get(token) {
                *out = *v;
                continue;
            }
            let lowercase = token.to_lowercase();
            if let Some(v) = embedding.get(&lowercase) {
                *out = *v;
                continue;
            }

            let fenced = lowercase.replace(|c: char| c.is_ascii_digit(), "#");
            if let Some(v) = embedding.get(&fenced) {
                *out = *v;
                continue;
            }

            let nulled = lowercase.replace(|c: char| c.is_ascii_digit(), "0");
            if let Some(v) = embedding.get(&nulled) {
                *out = *v;
                continue;
            }
            
            *out = Vector::null();
        }
    }
    pub fn viterbi_decode(&self, features: &[Vector<20>], out: &mut [u8]) {
        let id_start = self.tags.iter().position(|t| &**t == "<START>").unwrap();
        let id_stop = self.tags.iter().position(|t| &**t == "<STOP>").unwrap();

        let mut backpointers = vec![[0; FEATURES]; features.len()];
        let mut backscores = vec![Vector::null(); features.len()];

        let mut forward_var = Vector::splat(-1e4);
        forward_var.set(id_start, 0.0);

        for (index, feat) in features.iter().enumerate() {
            let (viterbivars_t, bptrs_t) = self.transitions.argmax1(&forward_var);
            forward_var = viterbivars_t + feat;
            backscores[index] = forward_var;
            backpointers[index] = bptrs_t;
        }

        let mut terminal_var = forward_var + self.transitions[id_stop];
        terminal_var[id_stop] = -1e5;
        terminal_var[id_start] = -1e5;

        let (mut best_tag_id, _) = terminal_var.max_idx();
        let mut best_path = vec![best_tag_id];
        for bptrs_t in backpointers.iter().rev() {
            best_tag_id = bptrs_t[best_tag_id];
            best_path.push(best_tag_id);
        }

        let start = best_path.pop().unwrap();
        assert_eq!(start, id_start);
        out.iter_mut().zip(best_path.iter().rev()).for_each(|(o, i)| *o = *i as u8);
    }

}
