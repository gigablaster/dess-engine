// Copyright (C) 2023 gigablaster

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use std::{hash::Hash, marker::PhantomData};

const DEFAULT_SPACE: usize = 4096;

#[derive(Debug)]
pub struct Handle<T, U>
where
    T: Copy,
{
    index: u32,
    generation: u32,
    _phantom1: PhantomData<T>,
    _phantom2: PhantomData<U>,
}

unsafe impl<T: Copy, U> Send for Handle<T, U> {}
unsafe impl<T: Copy, U> Sync for Handle<T, U> {}

#[allow(clippy::incorrect_clone_impl_on_copy_type)]
impl<T, U> Clone for Handle<T, U>
where
    T: Copy,
{
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            generation: self.generation,
            _phantom1: PhantomData,
            _phantom2: PhantomData,
        }
    }
}

impl<T, U> Copy for Handle<T, U> where T: Copy {}

impl<T, U> PartialEq for Handle<T, U>
where
    T: Copy,
{
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T, U> Eq for Handle<T, U> where T: Copy {}

impl<T, U> Hash for Handle<T, U>
where
    T: Copy,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T, U> Handle<T, U>
where
    T: Copy,
{
    pub fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            _phantom1: PhantomData,
            _phantom2: PhantomData,
        }
    }

    pub fn invalid() -> Self {
        Self {
            index: u32::MAX,
            generation: u32::MAX,
            _phantom1: PhantomData,
            _phantom2: PhantomData,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.index != u32::MAX && self.generation != u32::MAX
    }

    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }
}

impl<T, U> Default for Handle<T, U>
where
    T: Copy,
{
    fn default() -> Self {
        Self::invalid()
    }
}

pub struct HandleContainer<T, U>
where
    T: Copy,
{
    hot: Vec<T>,
    cold: Vec<Option<U>>,
    generations: Vec<u32>,
    empty: Vec<u32>,
}

impl<T, U> HandleContainer<T, U>
where
    T: Copy,
{
    pub fn new() -> Self {
        Self {
            hot: Vec::with_capacity(DEFAULT_SPACE),
            cold: Vec::with_capacity(DEFAULT_SPACE),
            generations: Vec::with_capacity(DEFAULT_SPACE),
            empty: Vec::with_capacity(DEFAULT_SPACE),
        }
    }

    pub fn push(&mut self, hot: T, cold: U) -> Handle<T, U> {
        if let Some(slot) = self.empty.pop() {
            self.hot[slot as usize] = hot;
            self.cold[slot as usize] = Some(cold);
            Handle::new(slot, self.generations[slot as usize])
        } else {
            let index = self.generations.len();
            if index == u32::MAX as _ {
                panic!("Too many items in HandleContainer.");
            }
            self.generations.push(0);
            self.hot.push(hot);
            self.cold.push(Some(cold));
            assert_eq!(self.generations.len(), self.hot.len());
            Handle::new(index as u32, 0)
        }
    }

    pub fn get(&self, handle: Handle<T, U>) -> Option<(&T, &U)> {
        if self.is_handle_valid(handle) {
            let index = handle.index() as usize;
            Some((&self.hot[index], self.cold[index].as_ref().unwrap()))
        } else {
            None
        }
    }

    pub fn get_hot(&self, handle: Handle<T, U>) -> Option<&T> {
        if self.is_handle_valid(handle) {
            let index = handle.index() as usize;
            Some(&self.hot[index])
        } else {
            None
        }
    }

    pub fn get_cold(&self, handle: Handle<T, U>) -> Option<&U> {
        if self.is_handle_valid(handle) {
            let index = handle.index() as usize;
            Some(self.cold[index].as_ref().unwrap())
        } else {
            None
        }
    }

    pub fn replace(&mut self, handle: Handle<T, U>, hot: T, cold: U) -> Option<(T, U)> {
        if self.is_handle_valid(handle) {
            let index = handle.index() as usize;
            let old_hot = self.hot[index];
            self.hot[index] = hot;
            let old_cold = self.cold[index].replace(cold).unwrap();

            Some((old_hot, old_cold))
        } else {
            None
        }
    }

    pub fn replace_hot(&mut self, handle: Handle<T, U>, hot: T) -> Option<T> {
        if self.is_handle_valid(handle) {
            let index = handle.index() as usize;
            let old_hot = self.hot[index];
            self.hot[index] = hot;

            Some(old_hot)
        } else {
            None
        }
    }

    pub fn replace_cold(&mut self, handle: Handle<T, U>, cold: U) -> Option<U> {
        if self.is_handle_valid(handle) {
            let index = handle.index() as usize;

            Some(self.cold[index].replace(cold).unwrap())
        } else {
            None
        }
    }

    pub fn remove(&mut self, handle: Handle<T, U>) -> Option<(T, U)> {
        if self.is_handle_valid(handle) {
            let index = handle.index as usize;
            self.generations[index] = self.generations[index].wrapping_add(1);
            self.empty.push(index as _);
            return Some((self.hot[index], self.cold[index].take().unwrap()));
        }

        None
    }

    pub fn is_handle_valid(&self, handle: Handle<T, U>) -> bool {
        let index = handle.index as usize;
        index < self.generations.len() && self.generations[index] == handle.generation()
    }
}

impl<T, U> Default for HandleContainer<T, U>
where
    T: Copy,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {
    use crate::{Handle, HandleContainer};

    #[test]
    fn handle() {
        let handle = Handle::<(), ()>::new(100, 10);
        assert_eq!(100, handle.index());
        assert_eq!(10, handle.generation());
    }

    #[test]
    fn handle_container_push_get() {
        let mut container = HandleContainer::<u32, i32>::new();
        let handle1 = container.push(1, -1);
        let handle2 = container.push(2, -2);
        let handle3 = container.push(3, -3);
        assert_eq!(Some((&1, &-1)), container.get(handle1));
        assert_eq!(Some((&2, &-2)), container.get(handle2));
        assert_eq!(Some((&3, &-3)), container.get(handle3));
        assert_eq!(Some(&2), container.get_hot(handle2));
        assert_eq!(Some(&-3), container.get_cold(handle3));
    }

    #[test]
    fn reuse_slot() {
        let mut container = HandleContainer::<u32, i32>::new();
        let handle = container.push(1, -1);
        container.remove(handle);
        let handle = container.push(2, -2);
        assert_eq!(1, handle.generation());
        assert_eq!(0, handle.index());
        assert_eq!(Some((&2, &-2)), container.get(handle));
    }

    #[test]
    fn old_handle_returns_none() {
        let mut container = HandleContainer::<u32, i32>::new();
        let handle1 = container.push(1, -1);
        assert_eq!(Some((1, -1)), container.remove(handle1));
        let handle2 = container.push(2, -2);
        assert_eq!(None, container.get(handle1));
        assert_eq!(Some((&2, &-2)), container.get(handle2));
    }

    #[test]
    fn mutate_by_handle() {
        let mut container = HandleContainer::<u32, i32>::new();
        let handle = container.push(1, -1);
        assert_eq!(Some((&1, &-1)), container.get(handle));
        assert_eq!(Some((1, -1)), container.replace(handle, 2, -2));
        assert_eq!(Some((&2, &-2)), container.get(handle));
        assert_eq!(Some(2), container.replace_hot(handle, 3));
        assert_eq!(Some(-2), container.replace_cold(handle, -3));
        assert_eq!(Some((&3, &-3)), container.get(handle));
    }
}
