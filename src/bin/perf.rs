#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

use nn::test::*;

fn main() {
    let mut sample = Lstm100x1024::new();
    for _ in 0 .. 5_000 {
        sample.step();
    }
}
