use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
}

#[derive(Debug)]
pub struct Worker {
    id: String,
    address: String,
    healthy: AtomicBool,
    counter: AtomicUsize,
}

pub struct InFlightGuard<'a> {
    worker: &'a Worker,
}

impl<'a> Drop for InFlightGuard<'a> {
    fn drop(&mut self) {
        self.worker.decrement_count();
    }
}

impl Worker {
    pub fn new(id: String, address: String) -> Self {
        Self {
            id,
            address,
            healthy: AtomicBool::new(true),
            counter: AtomicUsize::new(0),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    pub fn counter(&self) -> usize {
        self.counter.load(Ordering::Relaxed)
    }

    pub fn begin_request(&self) -> InFlightGuard<'_> {
        self.increment_count();
        InFlightGuard { worker: self }
    }

    fn increment_count(&self) {
        self.counter.fetch_add(1, Ordering::Relaxed);
    }

    fn decrement_count(&self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn status(&self) -> HealthStatus {
        if self.healthy.load(Ordering::Relaxed) {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unhealthy
        }
    }

    pub fn set_status(&self, status: HealthStatus) {
        let healthy = match status {
            HealthStatus::Healthy => true,
            HealthStatus::Unhealthy => false,
        };

        self.healthy.store(healthy, Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::{HealthStatus, Worker};

    #[test]
    fn counts_requests_per_worker() {
        let worker_a = Worker::new("worker-a".to_string(), "127.0.0.1:3212".to_string());
        let worker_b = Worker::new("worker-b".to_string(), "127.0.0.1:3213".to_string());

        assert_eq!(worker_a.counter(), 0);
        assert_eq!(worker_b.counter(), 0);

        {
            let _guard = worker_a.begin_request();
            assert_eq!(worker_a.counter(), 1);
            assert_eq!(worker_b.counter(), 0);
        }

        assert_eq!(worker_a.counter(), 0);
    }

    #[test]
    fn new_worker_is_healthy() {
        let worker = Worker::new(String::from("worker-a"), "127.0.0.1:3212".to_string());

        assert_eq!(worker.id(), "worker-a");
        assert_eq!(worker.status(), HealthStatus::Healthy);
    }

    #[test]
    fn worker_status_can_change() {
        let worker = Worker::new(String::from("worker-a"), "127.0.0.1:3212".to_string());

        worker.set_status(HealthStatus::Unhealthy);
        assert_eq!(worker.status(), HealthStatus::Unhealthy);

        worker.set_status(HealthStatus::Healthy);
        assert_eq!(worker.status(), HealthStatus::Healthy);
    }

    #[test]
    fn shares_health_status_between_arc_owners() {
        let worker = std::sync::Arc::new(Worker::new(
            String::from("worker-a"),
            "127.0.0.1:3212".to_string(),
        ));
        let another_owner = std::sync::Arc::clone(&worker);

        assert_eq!(worker.status(), HealthStatus::Healthy);

        worker.set_status(HealthStatus::Unhealthy);
        assert_eq!(another_owner.status(), HealthStatus::Unhealthy);

        another_owner.set_status(HealthStatus::Healthy);
        assert_eq!(worker.status(), HealthStatus::Healthy);
    }
}
