pub trait Splits {
    type Iter<I: Iterator<Item=usize>>: Iterator<Item=Self>;
    fn splits<I: Iterator<Item=usize>>(self, iter: I) -> Self::Iter<I>;
}
impl<'a, T> Splits for &'a [T] {
    type Iter<I: Iterator<Item=usize>> = impl Iterator<Item=&'a [T]>;
    
    fn splits<I: Iterator<Item=usize>>(self, iter: I) -> Self::Iter<I> {
        iter.scan(self, |slice, n| {
            if n > slice.len() {
                None
            } else {
                let (a, b) = slice.split_at(n);
                *slice = b;
                
                Some(a)
            }
        })
    }
}
impl<'a, T> Splits for &'a mut [T] {
    type Iter<I: Iterator<Item=usize>> = impl Iterator<Item=&'a mut [T]>;
    
    fn splits<I: Iterator<Item=usize>>(self, iter: I) -> Self::Iter<I> {
        iter.scan(self, |slice, n| {
            if n > slice.len() {
                None
            } else {
                let (a, b) = std::mem::replace(slice, &mut []).split_at_mut(n);
                *slice = b;
                
                Some(a)
            }
        })
    }
}
impl<'a, T, U> Splits for (&'a [T], &'a [U]) {
    type Iter<I: Iterator<Item=usize>> = impl Iterator<Item=(&'a [T], &'a [U])>;
    
    fn splits<I: Iterator<Item=usize>>(self, iter: I) -> Self::Iter<I> {
        iter.scan(self, |(slice_a, slice_b), n| {
            if n > slice_a.len() || n > slice_b.len() {
                None
            } else {
                let (a_head, a_tail) = slice_a.split_at(n);
                let (b_head, b_tail) = slice_b.split_at(n);
                *slice_a = a_tail;
                *slice_b = b_tail;
                
                Some((a_head,b_head))
            }
        })
    }
}
impl<'a, T, U, V> Splits for (&'a [T], &'a [U], &'a [V]) {
    type Iter<I: Iterator<Item=usize>> = impl Iterator<Item=(&'a [T], &'a [U], &'a [V])>;
    
    fn splits<I: Iterator<Item=usize>>(self, iter: I) -> Self::Iter<I> {
        iter.scan(self, |(slice_a, slice_b, slice_c), n| {
            if n > slice_a.len() || n > slice_b.len() || n > slice_c.len() {
                None
            } else {
                let (a_head, a_tail) = slice_a.split_at(n);
                let (b_head, b_tail) = slice_b.split_at(n);
                let (c_head, c_tail) = slice_c.split_at(n);
                *slice_a = a_tail;
                *slice_b = b_tail;
                *slice_c = c_tail;
                
                Some((a_head, b_head, c_head))
            }
        })
    }
}
