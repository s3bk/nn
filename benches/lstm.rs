#![feature(test)]
#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

extern crate test;
use test::Bencher;
use nn::test::*;

#[bench]
fn bench_lstm_100x1024(bencher: &mut Bencher) {
    let mut sample = Lstm100x1024::new();
    bencher.iter(|| sample.step())
}
