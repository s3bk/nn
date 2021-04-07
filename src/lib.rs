#![feature(const_generics, const_evaluatable_checked)]
#![feature(new_uninit)]
#![feature(type_alias_impl_trait)]
#![feature(generic_associated_types)]
#![allow(incomplete_features)]

#[macro_use] extern crate itertools;
#[macro_use] extern crate log;

pub use simd_linalg as vector;
pub mod lstm;
pub mod data;
pub mod test;
pub mod ner;
mod utils;

#[cfg(feature="gpu")]
pub mod gpu;
