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

use std::{
    collections::{hash_map::Entry, HashMap},
    sync::Mutex,
    thread::{self, ThreadId},
};

use ash::vk;

use crate::{BackendError, GpuResource};

use super::{CommandBuffer, CommandPool, Semaphore};

pub struct FrameContext {
    pub(crate) main_cb: CommandBuffer,
    pub(crate) presentation_cb: CommandBuffer,
    pub(crate) render_finished: Semaphore,
    secondary_pools: Mutex<HashMap<ThreadId, CommandPool>>,
    main_pool: CommandPool,
    query: u32,
}

unsafe impl Sync for FrameContext {}
unsafe impl Send for FrameContext {}

impl FrameContext {
    pub(crate) fn new(device: &ash::Device, query: u32) -> Result<Self, BackendError> {
        let render_finished = Semaphore::new(device)?;
        let command_pools = Mutex::new(HashMap::new());
        let mut main_pool = CommandPool::new(
            device,
            query,
            vk::CommandPoolCreateFlags::TRANSIENT,
            vk::CommandBufferLevel::PRIMARY,
        )?;
        let main_cb = main_pool.get_or_create(device)?;
        let presentation_cb = main_pool.get_or_create(device)?;

        Ok(Self {
            secondary_pools: command_pools,
            render_finished,
            query,
            main_pool,
            main_cb,
            presentation_cb,
        })
    }

    pub(crate) fn reset(&self, device: &ash::Device) -> Result<(), BackendError> {
        let mut pools = self.secondary_pools.lock();
        for (_, pool) in pools.iter_mut() {
            pool.recycle();
            pool.reset(device)?;
        }
        self.main_pool.reset(device)
    }

    pub fn get_or_create_secondary_command_buffer(
        &self,
        device: &ash::Device,
    ) -> Result<CommandBufferGuard, BackendError> {
        let thread_id = thread::current().id();
        let mut pools = self.secondary_pools.lock().unwrap();
        if let Entry::Vacant(e) = pools.entry(thread_id) {
            let pool = CommandPool::new(
                device,
                self.query,
                vk::CommandPoolCreateFlags::TRANSIENT,
                vk::CommandBufferLevel::SECONDARY,
            )?;
            e.insert(pool);
        }
        Ok(CommandBufferGuard {
            frame: self,
            cb: pools.get_mut(&thread_id).unwrap().get_or_create(device)?,
        })
    }

    pub(self) fn recycle_command_buffer(&self, cb: CommandBuffer) {
        let thread_id = thread::current().id();
        let mut pools = self.secondary_pools.lock().unwrap();
        pools
            .get_mut(&thread_id)
            .expect("Command buffer must be recycled from same thread it was allocated from")
            .retire(cb);
    }

    pub fn main_cb(&self) -> &CommandBuffer {
        &self.main_cb
    }
}

pub struct CommandBufferGuard<'a> {
    frame: &'a FrameContext,
    cb: CommandBuffer,
}

impl<'a> AsRef<CommandBuffer> for CommandBufferGuard<'a> {
    fn as_ref(&self) -> &CommandBuffer {
        &self.cb
    }
}

impl<'a> Drop for CommandBufferGuard<'a> {
    fn drop(&mut self) {
        self.frame.recycle_command_buffer(self.cb);
    }
}

impl GpuResource for FrameContext {
    fn free(&self, device: &ash::Device) {
        self.render_finished.free(device);
        let mut pools = self.secondary_pools.lock().unwrap();
        pools.iter_mut().for_each(|(_, pool)| pool.free(device));
        pools.clear();
        self.main_cb.free(device);
        self.presentation_cb.free(device);
        self.main_pool.free(device);
    }
}
