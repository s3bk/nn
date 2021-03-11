use crate::data::*;
use crate::vector::*;
use crate::lstm::*;
use unicode_segmentation::UnicodeSegmentation;
use itertools::Itertools;
use std::time::Instant;

pub struct NerTagger {
    model: Model
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

impl NerTagger {
    pub fn new(model: Model) -> Self {
        NerTagger { model }
    }
    pub fn tag_name(&self, tag: Tag) -> &str {
        &*self.model.tags[tag.0 as usize]
    }
    pub fn tag_by_name(&self, name: &str) -> Option<Tag> {
        self.model.tags.iter().position(|n| &**n == name).map(|n| Tag(n as u8))
    }
    pub fn tag(&self, text: &str) -> Vec<Token> {
        let (tokens_indices, tokens): (Vec<_>, Vec<_>) = self.tokenize(text).unzip();

        let mut glove_v = vec![Vector::null(); tokens.len()];
        let mut forward_v = vec![Vector::null(); tokens.len()];
        let mut reverse_v = vec![Vector::null(); tokens.len()];

        self.glove_embedding(&tokens, &mut glove_v);
        self.forward_embedding(&tokens, &mut forward_v);
        self.reverse_embedding(&tokens, &mut reverse_v);

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
        self.viterbi_decode(&features, &mut tags);
        izip!(tokens_indices, tokens, tags).map(|(pos, token, tag_id)|
            Token { pos, len: token.len(), tag: Tag(tag_id) }
        ).collect()
    }
    pub fn tag_par<'a, I, T>(&'a self, mut input: I, config: ParallelConfig) -> impl Iterator<Item=Tokens<T>> + 'a
    where I: Iterator<Item=T> + 'a, T: AsRef<str> + 'a
    {
        use rayon::prelude::*;
        info!("running batches of ~{} bytes in {} queues", config.batch_size, config.num_queues);

        let mut batch_nr = 0;
        std::iter::from_fn(move || {
            let mut batch_input = vec![];
            let mut queues = vec![(0, vec![]); config.num_queues];
            let mut total = 0;
            let t0 = Instant::now();

            while let Some(input) = input.next() {
                let text = input.as_ref();
                if text.len() == 0 {
                    continue;
                }
                total += text.len();
                batch_input.push(input);

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
            for (idx, input) in batch_input.iter().enumerate() {
                let (n, min_q) = queues.iter_mut().min_by_key(|(n, _)| *n).unwrap();
                let text = input.as_ref();
                min_q.push((idx, text));
                *n += text.len();
            }
            
            //println!("queues {:#?}", &queues);
            info!("queue sizes: {}", queues.iter().map(|(_, q)| q.len()).format(", "));
            info!("queue lenghts: {}", queues.iter().map(|(n, _)| n).format(", "));

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

            // tokenize each queue
            info!("tokenizing");
            let mut token_queues: Vec<_> = queues.par_iter().map(|(_, queue)| {
                let mut tokens = vec![];
                let mut offsets = vec![];
                let mut meta = vec![];
                let mut num_tokens = 0;
                for &(input_id, text) in queue {
                    let mut text_tokens = 0;
                    for (token_offset, token) in self.tokenize(text) {
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

            // glove pass
            info!("glove embeddings");
            token_queues.par_iter_mut().for_each(|queue| {
                queue.glove.resize(queue.num_tokens, Vector::null());
                let mut start = 0;
                for &(_, num_tokens) in &queue.meta {
                    let end = start + num_tokens;
                    self.glove_embedding(&queue.tokens[start..end], &mut queue.glove[start..end]);
                    start = end;
                }
            });

            // forward pass
            info!("forward char embeddings");
            token_queues.par_iter_mut().for_each(|q| {
                q.forward.resize(q.num_tokens, Vector::null());
                let mut start = 0;
                for &(_, num_tokens) in &q.meta {
                    let end = start + num_tokens;
                    self.forward_embedding(&q.tokens[start..end], &mut q.forward[start..end]);
                    start = end;
                }
            });

            // reverse pass
            info!("reverse char embeddings");
            token_queues.par_iter_mut().for_each(|q| {
                q.reverse.resize(q.num_tokens, Vector::null());
                let mut start = 0;
                for &(_, num_tokens) in &q.meta {
                    let end = start + num_tokens;
                    self.reverse_embedding(&q.tokens[start..end], &mut q.reverse[start..end]);
                    start = end;
                }
            });

            // transform embeddings
            info!("transforming embeddings");
            token_queues.par_iter_mut().for_each(|q| {
                q.embeddings.resize(q.num_tokens, Vector::null());

                for (forward, reverse, glove, out) in izip!(&q.forward, &q.reverse, &q.glove, &mut q.embeddings) {
                    let embedding = glove.concat2(&reverse, &forward);
                    *out = self.model.embedding2nn.transform(&embedding);
                }
                q.glove = vec![];
                q.forward = vec![];
                q.reverse = vec![];
            });

            // RNN forward
            info!("forward RNN");
            token_queues.par_iter_mut().for_each(|q| {
                q.rnn_forward.resize(q.num_tokens, Vector::null());

                let mut start = 0;
                for &(_, num_tokens) in &q.meta {
                    let end = start + num_tokens;
                    let mut state_forward = LstmState::null();
                    let embedding = &q.embeddings[start..end];
                    let rnn_forward = &mut q.rnn_forward[start..end];

                    for (embedding, out) in embedding.iter().zip(rnn_forward.iter_mut()) {
                        self.model.rnn.step(&mut state_forward, embedding);
                        *out = state_forward.h;
                    }

                    start = end;
                }
            });

            // RNN reverse
            info!("reverse RNN");
            token_queues.par_iter_mut().for_each(|q| {
                q.rnn_reverse.resize(q.num_tokens, Vector::null());
                let mut start = 0;
                for &(_, num_tokens) in &q.meta {
                    let end = start + num_tokens;
                    let mut state_reverse = LstmState::null();
                    let embedding = &q.embeddings[start..end];
                    let rnn_reverse = &mut q.rnn_reverse[start..end];

                    for (embedding, out) in embedding.iter().zip(rnn_reverse.iter_mut()).rev() {
                        self.model.rnn_reverse.step(&mut state_reverse, embedding);
                        *out = state_reverse.h;
                    }

                    start = end;
                }
                q.embeddings = vec![];
            });
            
            // feature transform
            info!("transforming features");
            token_queues.par_iter_mut().for_each(|q| {
                q.features.resize(q.num_tokens, Vector::null());
                for (out, forward, reverse) in izip!(&mut q.features, &q.rnn_forward, &q.rnn_reverse) {
                    *out = self.model.linear.transform(&forward.concat(reverse));
                }
                q.rnn_forward = vec![];
                q.rnn_reverse = vec![];
            });

            // tag extraction
            info!("extracting tags");
            token_queues.par_iter_mut().for_each(|q| {
                q.tags.resize(q.num_tokens, 0);

                let mut start = 0;
                for &(_, num_tokens) in &q.meta {
                    let end = start + num_tokens;
                    self.viterbi_decode(&q.features[start..end], &mut q.tags[start..end]);

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

            Some(groups.into_iter().zip(batch_input.into_iter().enumerate())
                .map(|((idx1, tokens), (idx2, input))| {
                    assert_eq!(idx1, idx2);
                    Tokens { input, tokens }
                })
            )
        }).flatten()
    }
    fn forward_embedding(&self, tokens: &[&str], out: &mut[Vector<FORWARD_EMBEDDING>]) {
        let embedding = &self.model.embeddings_forward;
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

    fn reverse_embedding(&self, tokens: &[&str], out: &mut [Vector<REVERSE_EMBEDDING>]) {
        let embedding = &self.model.embeddings_reverse;
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
    fn glove_embedding(&self, tokens: &[&str], out: &mut [Vector<GLOVE_EMBEDDING>]) {
        let embedding = &self.model.embeddings_glove;

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
    fn viterbi_decode(&self, features: &[Vector<20>], out: &mut [u8]) {
        let id_start = self.model.tags.iter().position(|t| &**t == "<START>").unwrap();
        let id_stop = self.model.tags.iter().position(|t| &**t == "<STOP>").unwrap();

        let mut backpointers = vec![[0; FEATURES]; features.len()];
        let mut backscores = vec![Vector::null(); features.len()];

        let mut forward_var = Vector::splat(-1e4);
        forward_var.set(id_start, 0.0);

        for (index, feat) in features.iter().enumerate() {
            let (viterbivars_t, bptrs_t) = self.model.transitions.argmax1(&forward_var);
            forward_var = viterbivars_t + feat;
            backscores[index] = forward_var;
            backpointers[index] = bptrs_t;
        }

        let mut terminal_var = forward_var + self.model.transitions[id_stop];
        terminal_var[id_stop] = -1e5;
        terminal_var[id_start] = -1e5;

        let (mut best_tag_id, _) = terminal_var.max_idx();
        let mut best_path = vec![best_tag_id];
        for bptrs_t in backpointers.iter().rev() {
            best_tag_id = bptrs_t[best_tag_id];
            best_path.push(best_tag_id);
        }

        let start = best_path.pop().unwrap();
        if start != id_start {
            error!("expected <START>, found {}", self.tag_name(Tag(start as u8)));
        }
        out.iter_mut().zip(best_path.iter().rev()).for_each(|(o, i)| *o = *i as u8);
    }

    fn tokenize<'a>(&'a self, text: &'a str) -> impl Iterator<Item=(usize, &'a str)> + 'a
    {
        text.split_word_bound_indices()
        .filter(|(_, s)| s.len() > 0 && !s.chars().all(char::is_whitespace))
    }
}