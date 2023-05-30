use std::{hash::Hash, marker::PhantomData, ptr::replace};

const GENERATOPN_OFFSET: u32 = 20;
const INDEX_MASK: u32 = (1 << GENERATOPN_OFFSET) - 1;
const GENERATON_MASK: u32 = u32::MAX & !INDEX_MASK;
const DEFAULT_SPACE: usize = 1024;

#[derive(Debug)]
pub struct Handle<T> {
    id: u32,
    _phantom: PhantomData<T>,
}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for Handle<T> {}

impl<T> Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T> Copy for Handle<T> {}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            _phantom: PhantomData::<T>,
        }
    }
}

impl<T> Handle<T> {
    pub fn new(index: u32, generation: u32) -> Self {
        assert!(index <= INDEX_MASK);
        assert!(generation <= GENERATON_MASK >> GENERATOPN_OFFSET);
        Self {
            id: (generation << GENERATOPN_OFFSET) | index,
            _phantom: PhantomData::<T>,
        }
    }

    pub fn invalid() -> Self {
        Self {
            id: u32::MAX,
            _phantom: PhantomData::<T>,
        }
    }

    pub fn index(&self) -> u32 {
        self.id & INDEX_MASK
    }

    pub fn generation(&self) -> u32 {
        (self.id & GENERATON_MASK) >> GENERATOPN_OFFSET
    }
}

impl<T> Default for Handle<T> {
    fn default() -> Self {
        Self::invalid()
    }
}

pub struct HandleContainer<T> {
    data: Vec<Option<T>>,
    generations: Vec<u32>,
    empty: Vec<u32>,
}

impl<T> HandleContainer<T> {
    pub fn new() -> Self {
        Self {
            data: Vec::with_capacity(DEFAULT_SPACE),
            generations: Vec::with_capacity(DEFAULT_SPACE),
            empty: Vec::with_capacity(DEFAULT_SPACE),
        }
    }

    pub fn push(&mut self, value: T) -> Handle<T> {
        if let Some(slot) = self.empty.pop() {
            self.data[slot as usize] = Some(value);
            Handle::new(slot, self.generations[slot as usize])
        } else {
            let index = self.generations.len();
            if index >= INDEX_MASK as _ {
                panic!("Too many items in HandleContainer.");
            }
            self.generations.push(0);
            self.data.push(Some(value));
            assert_eq!(self.generations.len(), self.data.len());
            Handle::new(index as u32, 0)
        }
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        let index = handle.index() as usize;
        if index < self.generations.len() && self.generations[index] == handle.generation() {
            if let Some(value) = &self.data[index] {
                return Some(value);
            }
        }

        None
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        let index = handle.index() as usize;
        if index < self.generations.len() && self.generations[index] == handle.generation() {
            if let Some(value) = &mut self.data[index] {
                return Some(value);
            }
        }

        None
    }

    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        let index = handle.index() as usize;
        if index < self.generations.len() && self.generations[index] == handle.generation() {
            self.generations[index] = self.generations[index].wrapping_add(1);
            self.empty.push(index as _);
            if let Some(value) = unsafe { replace(&mut self.data[index], None) } {
                return Some(value);
            }
        }

        None
    }
}

impl<T> Default for HandleContainer<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {
    use crate::{Handle, HandleContainer};

    #[test]
    fn handle() {
        let handle = Handle::<()>::new(100, 10);
        assert_eq!(100, handle.index());
        assert_eq!(10, handle.generation());
    }

    #[test]
    fn handle_container_push_get() {
        let mut container = HandleContainer::<u32>::new();
        let handle1 = container.push(1);
        let handle2 = container.push(2);
        let handle3 = container.push(3);
        assert_eq!(Some(&1), container.get(handle1));
        assert_eq!(Some(&2), container.get(handle2));
        assert_eq!(Some(&3), container.get(handle3));
    }

    #[test]
    fn reuse_slot() {
        let mut container = HandleContainer::<u32>::new();
        let handle = container.push(1);
        container.remove(handle);
        let handle = container.push(2);
        assert_eq!(1, handle.generation());
        assert_eq!(0, handle.index());
        assert_eq!(Some(&2), container.get(handle));
    }

    #[test]
    fn old_handle_returns_none() {
        let mut container = HandleContainer::<u32>::new();
        let handle1 = container.push(1);
        assert_eq!(Some(1), container.remove(handle1));
        let handle2 = container.push(2);
        assert_eq!(None, container.get(handle1));
        assert_eq!(Some(&2), container.get(handle2));
    }

    #[test]
    fn mutate_by_handle() {
        let mut container = HandleContainer::<u32>::new();
        let handle = container.push(1);
        assert_eq!(Some(&1), container.get(handle));
        *container.get_mut(handle).unwrap() = 2;
        assert_eq!(Some(&2), container.get(handle));
    }
}
