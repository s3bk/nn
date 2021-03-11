#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

use nn::data::Model;
use nn::ner::{NerTagger, ParallelConfig};
use std::fs::File;
use std::io::{BufReader, BufRead};
use argh::FromArgs;

#[derive(FromArgs)]
/// NER tagger options
struct FlairOpt {
    /// number of queues
    #[argh(option, default="16")]
    queues: usize,

    /// batch size in bytes
    #[argh(option, default="1024 * 16")]
    size: usize,

    /// model file
    #[argh(option, default="String::from(\"model.npz\")")]
    model: String,

    /// file to run
    #[argh(positional)]
    file: String
}

fn main() {
    env_logger::init();
    let opt = argh::from_env::<FlairOpt>();

    let file = File::open(&opt.model).unwrap();
    let model = Model::load(file).unwrap();
    let ner = NerTagger::new(model);


    let file = File::open(&opt.file).unwrap();
    let reader = BufReader::new(file);

    let par_opt = ParallelConfig {
        num_queues: opt.queues,
        batch_size: opt.size,
    };

    let other = ner.tag_by_name("O").unwrap();
    //let pers = ner.tag_by_name("O").unwrap();
    for tokens in ner.tag_par(reader.lines().filter_map(Result::ok), par_opt) {
        let para = tokens.input;
        for token in tokens.tokens {
            if token.tag != other {
            //    println!("{} : {}", &para[token.pos .. token.pos + token.len], ner.tag_name(token.tag));
            }
        }
    }
}
