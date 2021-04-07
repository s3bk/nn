use crate::data::*;
use crate::vector::*;
use crate::lstm::*;
use unicode_segmentation::UnicodeSegmentation;
use itertools::Itertools;
use std::time::Instant;
use crate::utils::*;
use rayon::prelude::*;
use std::ops::{Deref, DerefMut};
use std::iter::{once, repeat};
use std::marker::PhantomData;

#[cfg(feature="gpu")]
use cuda::{Device, Context, Module, CudaError};

pub struct NerTagger<'a> {
    model: Model,

    #[cfg(feature="gpu")]
    cuda: Option<(&'a Context, Module<'a>)>,

    #[cfg(not(feature="gpu"))]
    cuda: PhantomData<&'a ()>,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub pos: usize,
    pub len: usize,
    pub tag: Tag,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Tag(u8);

pub struct Tokens<I> {
    pub input: I,
    pub tokens: Vec<Token>,
}

pub struct ParallelConfig {
    pub num_queues: usize,
    pub batch_size: usize,
}
impl Default for ParallelConfig {
    fn default() -> Self {
        let num_queues = num_cpus::get_physical() * 4;
        ParallelConfig {
            num_queues,
            batch_size: num_queues * 1024,
        }
    }
}

//static MODULE_DATA: &[u8] = include_bytes!("../../ptx_linalg/ptx_linalg.cubin");

#[derive(Default)]
struct TokenQueue<'a> {
    num_tokens: usize,

    // (input_id, num_tokens)
    meta: Vec<(usize, usize)>,

    // shared across inputs
    tokens: Vec<&'a str>,
    offsets: Vec<usize>,
    glove: Vec<Vector<GLOVE_EMBEDDING>>,
    forward: Vec<Vector<FORWARD_EMBEDDING>>,
    reverse: Vec<Vector<REVERSE_EMBEDDING>>,
    embeddings: Vec<Vector<EMBEDDINGS>>,
    rnn_forward: Vec<Vector<RNN_SIZE>>,
    rnn_reverse: Vec<Vector<RNN_SIZE>>,
    features: Vec<Vector<FEATURES>>,
    tags: Vec<u8>,
}

macro_rules! pass {
    (iter $q:ident ($($input:ident),*), $output:ident, $pass:expr) => ({
        let input = ($( AsRef::<[_]>::as_ref(&$q.$input) ),* );
        let output = AsMut::<[_]>::as_mut(&mut $q.$output);
        let splits = $q.meta.iter().map(|&(_, num_tokens)| num_tokens);

        let input = input.splits(splits.clone());
        let output = output.splits(splits);
        #[allow(unused_parens)]
        input.zip(output).flat_map(|(($($input),*) , $output)| $pass)
    });
    (
        $queues:ident,
        $cuda_model:ident,
        input $($input:ident),* ;
        output $output:ident ;
        pass $pass:expr,
        cpu $cpu:expr,
        gpu $model:ident $gpu:expr,
    ) => ({
        #[cfg(feature="gpu")]
        if let Some(ref mut $model) = $cuda_model {
            let iter = $queues.iter_mut().map(|q| pass!(iter q ($($input),*), $output, $pass));
            $gpu(iter).unwrap();
        } else {
            $queues.par_iter_mut().for_each(|q| {
                let iter = pass!(iter q ($($input),*), $output, $pass);
                $cpu(iter)
            });
        }

        #[cfg(not(feature="gpu"))]
        $queues.par_iter_mut().for_each(|q| {
            let iter = pass!(iter q ($($input),*), $output, $pass);
            $cpu(iter)
        });
    });
}

impl<'a> NerTagger<'a> {
    pub fn new(model: Model) -> Self {
        NerTagger {
            model,
            #[cfg(not(feature="gpu"))]
            cuda: PhantomData,
            #[cfg(feature="gpu")]
            cuda: None,
        }
    }
    #[cfg(feature="gpu")]
    pub fn new_with_context(model: Model, context: &'a Context) -> Result<Self, CudaError> {
        let data = std::fs::read("../ptx_linalg/ptx_linalg.cubin").unwrap();
        let module = context.create_module(data)?;

        Ok(NerTagger { model, cuda: Some((context, module)) })
    }
    pub fn tag_name(&self, tag: Tag) -> &str {
        &*self.model.tags[tag.0 as usize]
    }
    pub fn tag_by_name(&self, name: &str) -> Option<Tag> {
        self.model.tags.iter().position(|n| &**n == name).map(|n| Tag(n as u8))
    }
    pub fn tag(&self, text: &str) -> Vec<Token> {
        let (tokens_indices, tokens): (Vec<_>, Vec<_>) = tokenize(text).unzip();

        let mut glove_v = vec![Vector::null(); tokens.len()];
        let mut forward_v = vec![Vector::null(); tokens.len()];
        let mut reverse_v = vec![Vector::null(); tokens.len()];

        self.model.glove_embedding(&tokens, &mut glove_v);
        self.model.forward_embedding(&tokens, &mut forward_v);
        self.model.reverse_embedding(&tokens, &mut reverse_v);

        let mut embeddings = Vec::with_capacity(tokens.len());

        for (forward, reverse, glove) in izip!(&forward_v, &reverse_v, &glove_v) {
            //println!("token: {}", &text[start..end]);
            let embedding: Vector<EMBEDDINGS> = glove.concat2(&reverse, &forward);
            let embedding = self.model.embedding2nn.transform(&embedding);
            //println!(" repojected: {}", &embedding);
            embeddings.push(embedding);
        }

        let mut rnn_out = vec![[Vector::null(); 2]; tokens.len()];

        let mut state_forward = LstmState::null();
        for (embedding, out) in embeddings.iter().zip(rnn_out.iter_mut()) {
            self.model.rnn.step(&mut state_forward, embedding);
            out[0] = state_forward.h;
        }

        let mut state_reverse = LstmState::null();
        for (embedding, out) in embeddings.iter().zip(rnn_out.iter_mut()).rev() {
            self.model.rnn_reverse.step(&mut state_reverse, embedding);
            out[1] = state_reverse.h;
        }

        let features: Vec<_> = rnn_out.iter()
            .map(|[forward, reverse]|
                self.model.linear.transform(&forward.concat(reverse))
            ).collect();

        let mut tags = vec![0; features.len()];
        self.model.viterbi_decode(&features, &mut tags);
        izip!(tokens_indices, tokens, tags).map(|(pos, token, tag_id)|
            Token { pos, len: token.len(), tag: Tag(tag_id) }
        ).collect()
    }
    pub fn tag_par<'b, I, T>(&'b self, mut input: I, config: ParallelConfig) -> impl Iterator<Item=Tokens<T>> + 'b
    where I: Iterator<Item=(usize, T)> + 'b, T: AsRef<str> + 'b
    {
        info!("running batches of ~{} bytes in {} queues", config.batch_size, config.num_queues);

        #[cfg(feature="gpu")]
        let mut cuda_model = self.cuda.as_ref().map(|&(context, ref module)| CudaModel::load(&self.model, context, module).unwrap());

        #[cfg(not(feature="gpu"))]
        let cuda_model = ();

        let mut batch_nr = 0;
        std::iter::from_fn(move || {
            let mut batch_input = vec![];
            let mut queues = vec![(0, vec![]); config.num_queues];
            let mut total = 0;
            let t0 = Instant::now();

            while let Some((idx, input)) = input.next() {
                let text = input.as_ref();
                if text.len() == 0 {
                    continue;
                }
                total += text.len();
                batch_input.push((idx, input));

                if total > config.batch_size {
                    break;
                }
            }
            if batch_input.len() == 0 {
                return None;
            }

            info!("batch nr. {} with {} inputs with a total size of {} bytes", batch_nr, batch_input.len(), total);

            // batch input is now stable and can be borrowed

            // distribute batch input to queues
            for (idx, input) in batch_input.iter() {
                let (n, min_q) = queues.iter_mut().min_by_key(|(n, _)| *n).unwrap();
                let text = input.as_ref();
                min_q.push((*idx, text));
                *n += text.len();
            }
            
            //println!("queues {:#?}", &queues);
            info!("queue sizes: {}", queues.iter().map(|(_, q)| q.len()).format(", "));
            info!("queue lenghts: {}", queues.iter().map(|(n, _)| n).format(", "));


            // tokenize each queue
            info!("tokenizing");
            let mut token_queues: Vec<_> = queues.par_iter().map(|(_, queue)| {
                let mut tokens = vec![];
                let mut offsets = vec![];
                let mut meta = vec![];
                let mut num_tokens = 0;
                for &(input_id, text) in queue {
                    let mut text_tokens = 0;
                    for (token_offset, token) in tokenize(text) {
                        if token.chars().all(char::is_whitespace) {
                            continue;
                        }
                        tokens.push(token);
                        offsets.push(token_offset);
                        text_tokens += 1;
                    }

                    meta.push((input_id, text_tokens));
                    num_tokens += text_tokens;
                }
                TokenQueue {
                    num_tokens,
                    tokens, offsets, meta,
                    .. Default::default()
                }
            }).collect();

            let model = &self.model;

            // glove pass
            info!("glove embeddings");
            token_queues.par_iter_mut().for_each(|queue| {
                queue.glove.resize(queue.num_tokens, Vector::null());
                let mut start = 0;
                for &(_, num_tokens) in &queue.meta {
                    let end = start + num_tokens;
                    model.glove_embedding(&queue.tokens[start..end], &mut queue.glove[start..end]);
                    start = end;
                }
            });

            let init_state = &LstmState::null();
            let embedding = &self.model.embeddings_forward;
            let start_marker = '\n';
            let end_marker = ' ';
            let v_start = &embedding.chars[&start_marker];
            let v_end = &embedding.chars[&end_marker];

            // forward pass
            info!("forward char embeddings");
            token_queues.iter_mut().for_each(|q| q.forward.resize(q.num_tokens, Vector::null()));

            pass!(token_queues, cuda_model,
                input tokens;
                output forward;
                pass {
                    let head = std::iter::once((Some(init_state), v_start, None));
                    let body = tokens.iter().zip(forward).flat_map(|(&token, out)| {
                        token.chars().filter_map(|c| embedding.chars.get(&c))
                            .map(|v| (None, v, None))
                            .chain(std::iter::once((None, v_end, Some(out))))
                    });
    
                    head.chain(body)
                },
                cpu |iter| model.embeddings_forward.rnn.run(iter),
                gpu model |iter| model.embeddings_forward.run_batched(iter),
            );

            // reverse pass
            info!("reverse char embeddings");
            token_queues.iter_mut().for_each(|q| q.reverse.resize(q.num_tokens, Vector::null()));

            pass!(token_queues, cuda_model,
                input tokens;
                output reverse;
                pass {
                    let head = std::iter::once((Some(init_state), v_start, None));
                    let body = tokens.iter().zip(reverse).rev().flat_map(|(&token, out)| {
                        token.chars().filter_map(|c| embedding.chars.get(&c))
                            .map(|v| (None, v, None))
                            .chain(std::iter::once((None, v_end, Some(out))))
                    });
    
                    head.chain(body)
                },
                cpu |iter| model.embeddings_reverse.rnn.run(iter),
                gpu model |iter| model.embeddings_reverse.run_batched(iter),
            );

            // transform embeddings
            info!("transforming embeddings");
            token_queues.iter_mut().for_each(|q| q.embeddings.resize(q.num_tokens, Vector::null()));

            pass!(token_queues, cuda_model,
                input glove, forward, reverse;
                output embeddings;
                pass {
                    izip!(glove.iter(), forward.iter(), reverse.iter())
                    .map(|(glove, forward, reverse)| glove.concat2(&reverse, &forward))
                    .zip(embeddings.iter_mut())
                },
                cpu |iter| model.embedding2nn.run(iter),
                gpu model |iter| model.embeddings2rnn.run_batched(iter),
            );

            // RNN forward
            info!("forward RNN");
            token_queues.iter_mut().for_each(|q| {
                q.rnn_forward.resize(q.num_tokens, Vector::null());
                q.rnn_reverse.resize(q.num_tokens, Vector::null());
            });

            let init = &LstmState::null();
            pass!(token_queues, cuda_model,
                input embeddings;
                output rnn_forward;
                pass {
                    let state = once(Some(init)).chain(repeat(None));
                    izip!(state, embeddings, rnn_forward.iter_mut().map(Some))
                },
                cpu |iter| model.rnn.run(iter),
                gpu model |iter| model.rnn.run_batched(iter),
            );

            // RNN reverse
            info!("reverse RNN");
            pass!(token_queues, cuda_model,
                input embeddings;
                output rnn_reverse;
                pass {
                    let state = once(Some(init)).chain(repeat(None));
                    state.zip(embeddings.iter().zip(rnn_reverse).rev())
                    .map(|(s, (i, o))| (s, i, Some(o)))
                },
                cpu |iter| model.rnn.run(iter),
                gpu model |iter| model.rnn.run_batched(iter),
            );
            
            // feature transform
            info!("transforming features");
            token_queues.iter_mut().for_each(|q| {
                q.features.resize(q.num_tokens, Vector::null());
            });
            pass!(token_queues, cuda_model,
                input rnn_forward, rnn_reverse;
                output features;
                pass {    
                    rnn_forward.iter().zip(rnn_reverse)
                    .map(|(forward, reverse)| forward.concat(reverse))
                    .zip(features)
                },
                cpu |iter| model.linear.run(iter),
                gpu model |iter| model.linear.run_batched(iter),
            );

            // tag extraction
            info!("extracting tags");
            token_queues.par_iter_mut().for_each(|q| {
                q.tags.resize(q.num_tokens, 0);

                let mut start = 0;
                for &(_, num_tokens) in &q.meta {
                    let end = start + num_tokens;
                    model.viterbi_decode(&q.features[start..end], &mut q.tags[start..end]);

                    start = end;
                }

                q.features = vec![];
            });

            info!("building output");
            let groups: Vec<_> = token_queues.iter()
                .map(|q| {
                    let mut start = 0;
                    q.meta.iter().flat_map(move |&(idx, len)| {
                        let end = start + len;
                        let iter = izip!(
                            &q.offsets[start..end],
                            &q.tokens[start..end],
                            &q.tags[start..end]
                        ).map(move |(&offset, &token, &tag)| (idx, Token {
                            pos: offset,
                            len: token.len(),
                            tag: Tag(tag)
                        }));
                        start = end;

                        iter
                    })
                })
                .kmerge_by(|a, b| a.0 < b.0)
                .group_by(|&(idx, _)| idx).into_iter()
                .map(|(idx, group)| (
                    idx,
                    group.into_iter().map(|(_, token)| token).collect::<Vec<Token>>()
                ))
                .collect();
            //println!("{:#?}", &groups);
            
            let seconds = t0.elapsed().as_secs_f32();
            info!("batch nr. {} is complete in {} seconds. ({} bytes/s)",
                batch_nr, seconds, total as f32 / seconds);
            batch_nr += 1;

            Some(groups.into_iter().zip(batch_input.into_iter())
                .map(|((idx1, tokens), (idx2, input))| {
                    assert_eq!(idx1, idx2);
                    Tokens { input, tokens }
                })
            )
        }).flatten()
    }
}

fn tokenize<'a>(text: &'a str) -> impl Iterator<Item=(usize, &'a str)> + 'a
{
    text.split_word_bound_indices()
    .filter(|(_, s)| s.len() > 0 && !s.chars().all(char::is_whitespace))
}