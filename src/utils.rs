use std::slice::from_raw_parts_mut;

pub(crate) fn split_at_mut_unchecked<T>(input: &mut [T], mid: usize) -> (&mut [T], &mut [T]) {
    let len = input.len();
    let ptr = input.as_mut_ptr();

    unsafe {
        assert!(mid <= len);

        (
            from_raw_parts_mut(ptr, mid),
            from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}
