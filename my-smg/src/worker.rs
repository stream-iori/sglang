#[derive(Debug, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
}

#[derive(Debug)]
pub struct Worker {
    id: String,
    status: HealthStatus,
}

impl Worker {
    pub fn new(id: String) -> Self {
        Self {
            id,
            status: HealthStatus::Healthy,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn status(&self) -> &HealthStatus {
        &self.status
    }

    pub fn set_status(&mut self, status: HealthStatus) {
        self.status = status;
    }
}

#[cfg(test)]
mod tests {
    use super::{HealthStatus, Worker};

    #[test]
    fn new_worker_is_healthy() {
        let worker = Worker::new(String::from("worker-a"));

        assert_eq!(worker.id(), "worker-a");
        assert_eq!(worker.status(), &HealthStatus::Healthy);
    }

    #[test]
    fn worker_status_can_change() {
        let mut worker = Worker::new(String::from("worker-a"));

        worker.set_status(HealthStatus::Unhealthy);
        assert_eq!(worker.status(), &HealthStatus::Unhealthy);

        worker.set_status(HealthStatus::Healthy);
        assert_eq!(worker.status(), &HealthStatus::Healthy);
    }
}
