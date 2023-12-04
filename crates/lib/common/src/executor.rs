use std::{future::Future, sync::Arc};

use async_task::{Runnable, Task};
use rayon::{ThreadPool, ThreadPoolBuilder};

/// Very dumb async executor that just spawns everythin on rayon pool
///
/// Pool is owned by executor.
pub struct Executor {
    pool: Arc<ThreadPool>,
}

impl Executor {
    pub fn spawn<'a, T, F>(&'a self, future: F) -> Task<T>
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static,
    {
        let pool = self.pool.clone();
        let (runnable, task) = async_task::spawn(future, move |runnable: Runnable| {
            pool.spawn(move || {
                runnable.run();
            })
        });

        runnable.schedule();

        task
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self {
            pool: Arc::new(ThreadPoolBuilder::default().build().unwrap()),
        }
    }
}

#[cfg(test)]
mod test {
    use std::{
        sync::{
            atomic::{AtomicU32, Ordering},
            Arc,
        },
        thread,
        time::Duration,
    };

    use crate::Executor;

    #[test]
    fn spawn_many_asyncs() {
        let executor = Executor::default();
        let counter = Arc::new(AtomicU32::new(0));
        let tasks = (0..64)
            .map(|_| executor.spawn(sleep(counter.clone())))
            .collect::<Vec<_>>();
        loop {
            if tasks.iter().all(|x| x.is_finished()) {
                break;
            }
        }
        assert_eq!(64, counter.load(Ordering::Acquire));
    }

    async fn sleep(n: Arc<AtomicU32>) {
        thread::sleep(Duration::from_millis(100));
        n.fetch_add(1, Ordering::AcqRel);
    }

    #[test]
    fn spawn_with_await() {
        let executor = Executor::default();
        let counter = Arc::new(AtomicU32::new(0));
        let tasks = (0..32)
            .map(|_| executor.spawn(sleep2(counter.clone())))
            .collect::<Vec<_>>();
        loop {
            if tasks.iter().all(|x| x.is_finished()) {
                break;
            }
        }
        assert_eq!(64, counter.load(Ordering::Acquire));
    }

    async fn sleep2(n: Arc<AtomicU32>) {
        sleep(n.clone()).await;
        sleep(n.clone()).await;
    }
}
