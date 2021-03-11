#![feature(const_generics, const_evaluatable_checked)]
#![feature(new_uninit)]
#![allow(incomplete_features)]

#[macro_use] extern crate itertools;
#[macro_use] extern crate log;

pub use simd_linalg as vector;
pub mod lstm;
pub mod data;
pub mod test;
pub mod ner;
